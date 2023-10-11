"""
Module that contains the command line app.
"""
import argparse
import os
from typing import List
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
import uuid
import vertexai
from vertexai.language_models import TextGenerationModel
import re
import random
import json
import jsonschema
import shutil
import traceback
import tensorflow as tf

from jsonschema import validate
from sklearn.model_selection import train_test_split
# from scikit-learn import train_test_split


DEFAULT_REGION = os.environ.get("GOOGLE_DEFAULT_REGION")
GOOGLE_BUCKET_NAME = os.environ.get("GOOGLE_BUCKET_NAME")
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LABELED_DIR = os.environ.get("LABELED_DIR")
DVC_DIR = "dvc_store"
TF_DATA_DIR = "TF_DATA"


def connect_to_bucket():
    """
    Connects to the storage bucket.
    """
    storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = storage_client.bucket(GOOGLE_BUCKET_NAME)
    return bucket


def show(guid=None):
    """
    Prints a training example with the given id. If no id is provided,
    it picks a random example from the list to show.
    """
    if guid is None:
        guids = list_impl()
        guid = random.choice(guids)
    path = f"{LABELED_DIR}/{guid}.json"
    print("Printing example " + path)

    bucket = connect_to_bucket()
    blob = bucket.blob(path)

    with blob.open("r") as f:
        print(f.read())


def list_impl() -> List[str]:
    """
    Returns the ids of all training examples in the by id.
    """
    guids = []
    guid_pattern = r"(\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)"
    for blob in connect_to_bucket().list_blobs(prefix=f"{LABELED_DIR}/"):
        match = re.search(guid_pattern, blob.name)
        if match:
            guids.append(match.group(0))
    return guids


def list() -> List[str]:
    """
    Prints the ids of all training examples by id.
    """
    print("Listing training examples:\n")
    guids = list_impl()
    for guid in guids:
        print(guid)


def setup():
    """
    Configures the bucket to be used for training.
    If the bucket does not already exist, it will be created.
    This will also check permissions are correct to read and write training examples
    to bucket.
    """

    client = storage.Client()
    bucket = Bucket(client, GOOGLE_BUCKET_NAME)
    if bucket.exists():
        print(f"Bucket '${GOOGLE_BUCKET_NAME}' arleady exists.")
    else:
        bucket.create(location=DEFAULT_REGION)

    # Create directories
    bucket.blob(LABELED_DIR + "/").upload_from_string("")
    bucket.blob(DVC_DIR + "/").upload_from_string("")


def create_and_upload_generated_data(n):
    """
    Sets up Vertex AI API and stores n generated training examples in bucket.
    """

    bucket = connect_to_bucket()

    # Model garden is not in all regions so hardcoded here
    vertexai.init(project=GOOGLE_CLOUD_PROJECT, location="us-central1")
    parameters = {
        "max_output_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")

    neg_goal = round(2*n/3)
    pos_goal = round(n/3)
    i_n = 0
    i_p = 0

    while i_n < neg_goal:
        contents = generate_neg(model, parameters)
        print(contents)
        if validate_json(contents):
            unique_id = str(uuid.uuid4())
            bucket.blob(LABELED_DIR + "/" + unique_id + ".json").upload_from_string(
                contents
            )
            print("Negative example added to bucket")
            i_n = i_n + 1

    while i_p < pos_goal:
        contents = generate_pos(model, parameters)
        print(contents)
        if validate_json(contents):
            unique_id = str(uuid.uuid4())
            bucket.blob(LABELED_DIR + "/" + unique_id + ".json").upload_from_string(
                contents
            )
            print("Positive example added to bucket")
            i_p = i_p + 1


def validate_json(example):
    json_schema = {
        "type": "object",
        "properties": {
            "record": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string"},
                    "received_time": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["sender", "received_time", "subject", "body"],
            },
            "label": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "subject": {"type": "string"},
                    },
                    "required": ["date", "time", "subject"],
                },
            },
        },
        "required": ["record", "label"]
    }

    try:
        validate(json.loads(example), json_schema)
    except jsonschema.exceptions.ValidationError as err:
        print("JSON Failed format")
        traceback.print_exc()
        return False
    except json.decoder.JSONDecodeError as err:
        print("Decoding issue. Probably new lines or quotes.")
        return False
    except:
        print("Not added, something else.")
        return False
    return True


def generate_neg(model, parameters):
    """
    Generates a training example given a prompt.
    """
    response = model.predict(
        """Generate a new json record where the record contains a random email without any mention of an event in the body.
            The sender field is the sender\'s name, received_time is the time the email was received, subject is the subject of the email, and body is the email text. 
            Make sure the sender, subject, and body are all different and unique. 
            Make the email body contain some random information and filler sentences. 
            For fields that expand to more than one line add an escaped newline character that can be read by Python ("\\n"). Put all the field "body" in one line.

        Use the examples below as a reference but also create your own. Make sure the JSON keys (record, sender, received_time, subject, body, label, date, time, and subject are in double quotes):
        {
        \"record\": {
                \"sender\" : \"Radical Ventures\",
                \"received_time\" : \"2023-04-21 6:00\",
                \"subject\": \"Thanks for joining!\",
                \"body\":\"Thank you for joining \\n Livestream: Geoffrey Hinton and Fei-Fei Li in conversation\\nWe hope you enjoyed the conversation between Geoffrey Hinton and Fei-Fei Li. ​This conversation was a part of the Radical AI Founders Masterclass program for AI researchers interested in commercializing their work. To learn more about the program, please visit: www.radical.vc\'
            },
            \"label\": []
        }

        {
        \"record\": {
                \"sender\" : \"News\",
                \"received_time\" : \"2023-04-21 6:00\",
                \"subject\": \"Chipotle robots to make burrito bowls / Rivian loses billions on electric truck quest / Lilly buys Point for $1.4B\",
                \"body\":\"Chipotle is conducting tests of a robot, developed in partnership with startup Hyphen, capable of assembling burrito bowls and salads exclusively for digital orders. Chipotle had invested in Hyphen, formerly known as Ono Food, in the previous year, and the startup currently holds a valuation of $104M. More:\\nAutomation strives to save labor costs and boost order accuracy and speed in the restaurant business.\\nChippy, a robot that creates tortilla chips, was previously used by Chipotle as an automation test subject.\\nThe Hyphen robot exclusively prepares burrito bowls and salads for digital orders, efficiently dispensing ingredients while allowing employees to assemble other digital orders like tacos, quesadillas, and burritos on the same line.\\nAccording to Chipotle, burrito bowls and salads account for almost two-thirds of all digital orders for the restaurant.\\nAnother restaurant brand, Sweetgreen, has already unveiled its first automated store and intends to automate every remaining establishment within five years.\"
            },
            \"label\": []
        }

        {
        \"record\": {
                \"sender\" : \"Tech Club\",
                \"received_time\" : \"2023-04-21 6:00\",
                \"subject\": \"Tech Club Newsletter #5 - Pizza Hangout | IBM & Microsoft Openings | The Cambridge Project\",
                \"body\":\"Hello TC Members Happy new month! This month is packed with activities, events, and meetups for our club, \\nand we cannot wait! Last week, it was great to see so many of you at our Bubble Tea Hangout where we all got to learn more about each other and strengthen our community. \\nICYMI, here is a picture from the event!\"
            },
            \"label\": []
        }

        {
        \"record\": {
                \"sender\" : \"IOP\",
                \"received_time\" : \"2023-04-21 6:00\",
                \"subject\": \"Rep. Joaquin Castro to address historic events in Congress, social and economic mobility for first generation Americans\",
                \"body\":\"In recognition of Hispanic Heritage Month, a conversation with U.S. Congressman Joaquin Castro (D-TX) and Alejandra Campoverdi, author of the FIRST GEN and former Obama White House official, as they unpack their own first generation journeys and explore the cost and achievability of social mobility for those who are the first in their families to cross a societal threshold.\"
            },
            \"label\": []
        }


        Please make sure the email does not invite the user any events.
        """,
        **parameters,
    )
    return response.text


def generate_pos(model, parameters):
    """
    Generates a training example given a prompt.
    """
    response = model.predict(
        """Generate a new json record where the record contains an email invitation to an event where the sender field is the sender\'s name, received_time is the time the email was received, subject is the subject of the email, and body is the email text. Label should contain information about the event like date, time, and subject of the event. Make sure the sender, subject, and body are all different and unique. Make the email body contain some random information and filler sentences. For fields that expand to more than one line add an escaped newline character that can be read by Python ("\\n"). Put all the field "body" in one line.
        Use the examples below as a reference and make sure the JSON keys (record, sender, received_time, subject, body, label, date, time, and subject are in double quotes):
        {
        \"record\": {
                \"sender\" : \"InTouch Coffee & Donuts - Wednesday at 3pm\",
                \"received_time\" : \"2023-04-21 6:00\",
                \"subject\": \"InTouch Coffee & Donuts - Wednesday at 3pm\",
                \"body\":\"Join us for donuts and coffee on Wednesday @ 3pm \\non the SEC 3rd floor (outside the grad student lounge)! \\nGrab a bite & meet fellow SEAS students :) \\n\\n- the InTouch team\\nimage.png\\n\\nWhat are InTouch coffee chats?\\nA safe space (with free food) for graduate students across SEAS to develop a broader sense of community, share in the highs and the lows. You don\'t need to have a \"problem\" to show up and chat.\\n\\nCheck out our well-stocked resources page and FAQs. \\nStay up to date with events by subscribing to the InTouch calendar.\"
            },
            \"label\": [{
                \"date\": \"April 7th 2021\",
                \"time\": \"3:00 PM\",
                \"subject\": \"InTouch Coffee & Donuts - Wednesday at 3pm\"
            }]
        }
        {
            \"record\": {
                \"sender\": \"John Smith\",
                \"received_time\": \"2023-03-08 10:00\",
                \"subject\": \"Event next week\",
                \"body\": \"Hi everyone,\\n\\nI\'m writing to let you know about an event that is taking place next week. The event is a fundraiser for the local animal shelter, and it will be held on March 15th at 7:00 PM at the local community center.\\n\\nWe are hoping to raise as much money as possible for the shelter, so we are asking everyone to come out and support the cause. There will be food, drinks, music, and a silent auction. Tickets are $20 per person, and children under 12 are free.\\n\\nWe hope to see you there!\\n\\nThanks,\\nJohn Smith\"
            },
            \"label\": [{
                \"date\": \"March 15th 2023\",
                \"time\": \"7:00 PM\",
                \"subject\": \"Local animal shelter fundraiser\"
            }]

        }
        {
        \"record\": {
            \"sender\": \"Alice Johnson\",
            \"received_time\": \"2023-09-25 15:30\",
            \"subject\": \"Join us for a Charity Gala!\",
            \"body\": \"Hello everyone,\\n\\nI hope this message finds you well. I\'m excited to invite you to our upcoming charity gala event in support of children\'s education.\\n\\nThe event will take place on October 10th, 2023, starting at 6:30 PM at the Grand Plaza Hotel. We have a fantastic evening planned with gourmet dining, live music, and inspiring speakers.\\n\\nYour presence would mean a lot to us as we aim to raise funds to provide educational resources to underprivileged children. Tickets are available at $50 per person, and we encourage you to bring your friends and family to join this noble cause.\\n\\nWe look forward to seeing you at the gala!\\n\\nBest regards,\\nAlice Johnson\"
        },
        \"label\": [
            {
            \"date\": \"October 10th 2023\",
            \"time\": \"6:30 PM\",
            \"subject\": \"Charity Gala for Children\'s Education\"
            }
        ]
        }
        {
        \"record\": {
            \"sender\": \"Devina Singh\",
            \"received_time\": \"2023-09-25 14:15\",
            \"subject\": \"Exclusive Wine Tasting Event\",
            \"body\": \"Dear wine enthusiasts,\\n\\n    I\'m thrilled to invite you to an exclusive wine tasting event that will take place on October 5th at 6:30 PM at the beautiful Vineyard Estates Winery. This event promises an unforgettable evening of fine wines and gourmet food pairings.\\n\\nYou\'ll have the opportunity to sample a selection of our finest wines and savor delectable dishes prepared by a renowned chef. Tickets are available for $50 per person and include a complimentary bottle of our special reserve wine.\\n\\nDon\'t miss this chance to indulge in a world of flavors and join fellow wine connoisseurs for an evening to remember.\\n\\nCheers,\\nDevina Singh\"
        },
        \"label\": [
            {
            \"date\": \"October 5th 2023\",
            \"time\": \"6:30 PM\",
            \"subject\": \"Vineyard Estates Wine Tasting\"
            }
        ]
        }
        {
        \"record\": {
            \"sender\": \"Jane Doe\",
            \"received_time\": \"2023-03-05 09:15\",
            \"subject\": \"Upcoming conference\",
            \"body\": \"Dear colleagues,\\n\\nI am writing to invite you to attend our upcoming regional conference on March 18th. The conference will be held at the Sheraton Hotel, beginning at 8am with registration and breakfast. \\\\nWe have an excellent lineup of speakers for this year\'s event covering topics such as new regulations, technology trends, and best practices. The $50 registration fee includes materials, lunch, and post-conference reception. \\n\\nPlease RSVP by March 10th if you plan to attend. I look forward to seeing many of you there!\\n\\nRegards,\\nJane Doe\"
        },

        \"label\": {
            \"date\": \"March 18th, 2023\",
            \"time\": \"8am\",
            \"subject\": \"Regional conference\"
        }
        }
        {
        \"record\": {
            \"sender\": \"Alice Johnson\",
            \"received_time\": \"2023-09-25 14:15\",
            \"subject\": \"Join us for the Annual Charity Run!\",
            \"body\": \"Hello everyone,\\n\\nI am thrilled to invite you to our Annual Charity Run happening on October 20th at 9:00 AM at City Park. This event is all about promoting fitness and raising funds for a great cause.\\n\\nThe run includes a 5K and 10K race, and participants of all levels are welcome. There will be refreshments, medals for top finishers, and a raffle with exciting prizes. Registration is open, and the entry fee is $25 per person.\\n\\nLet\'s lace up our running shoes and make a positive impact together!\\n\\nBest regards,\\nAlice Johnson\"
        },
        \"label\": [
            {
            \"date\": \"October 20th 2023\",
            \"time\": \"9:00 AM\",
            \"subject\": \"Annual Charity Run at City Park\"
            }
        ]
        }
        {
            \"record\": {
                \"sender\": \"Arthur Rock Center for Enterpreneurship\",
                \"received_time\": \"2023-09-2 6:00\",
                \"subject\": \"Meet with an Expert! Fall 2023 Rock Center Office Hours Begin This Week!\",
                \"body\": \"Rock Center\'s Expert Schedule\\nWednesday, September 27th \\nImage\\nSarah Leary \\nEntrepreneur-in-Residence\\nTime: 11:30 am - 2:30 pm ET\\nLocation: In-Person at the i-Lab\\nBook a time here\\n\\nSarah Leary, a seasoned technology entrepreneur, with expertise in B2C, team building, customer acquisition, and more across various industries. She co-founded Nextdoor, serving as VP of Marketing, International, and Operations, overseeing its global expansion. Currently a Venture Partner at Unusual Ventures, she leverages her extensive experience to benefit the consumer practice.\"
            },
            \"label\": [{
                \"date\": \'Sept 27th 2023\',
                \"time\": \'11:30 AM\',
                \"subject\": \'Meet with an Expert! Fall 2023 Rock Center Office Hours Begin This Week!\"
            }]
        }
        {
            \"record\": {
                \"sender\": \"Curren Iyer\",
                \"received_time\": \"2023-09-12 14:00\",
                \"subject\": \"Xfund x MSMBA 2024 Meet & Greet\",
                \"body\": \"Hi MS/MBAs!\\n\\nJoin us on Tuesday, Sept 26th from 5:15-6:15 pm to have your entrepreneurship questions answered by the experts at Xfund! \\n\\nXfund is the Harvard-born and based seed-stage VC firm that focuses on backing student entrepreneurs. The team has invested in companies like 23andMe, SpaceX, Robinhood, Kensho, and Patreon.\\n\\nThey are currently investing out of their $120m Xfund 3. \\nhttps://seas.harvard.edu/news/2020/09/xfund-celebrates-launch-new-fund \\n\\nMembers of the Xfund team -- Brandon (HBS \'14) and Jadyn (Harvard \'21, HBS \'26) -- will join us in Aldrich 210 to meet everyone and answer questions about anything on your mind as you think about entrepreneurship and the opportunities/resources available to you. \\n\\nDinner will be provided-- please RSVP on the invite so we know how much food to order.\\n\\nSee you there!\\n- Xfund team\"
            },
            \"label\": [{
                \"date\": \"Sept 26th 2023\",
                \"time\": \"5:15 PM\",
                \"subject\": \"Xfund x MSMBA 2024 Meet & Greet\"
            }]
        }
        {
            \"record\": {
                \"sender\": \"Mackenzie Lawrence\",
                \"received_time\": \"2022-05-1 10:00\",
                \"subject\": \"Ladies Sunset on the Terrace\",
                \"body\": \"Hi everyone,\\n\\nPlease RSVP! BYOB and BYOSnacks!\\n\\nI\'m hosting a ladies night on May 5. To get to the terrace, call MacKenzie on the call box, enter the building, pass first set of elevators and lounges to the second set of elevators, go to the second floor, and follow signs to terrace.\\n\\nI shouldn’t have forgotten anyone, but please forward along if I missed someone.\"
            },
            \"label\": [{
                \"date\": \"May 5th 2022\",
                \"time\": \"7:30 PM\",
                \"subject\": \"Ladies Sunset on the Terrace\"
            }]
        }

        """,
        **parameters,
    )
    # print(response.text)
    # res = list(map(str.strip, response.text.split(',')))
    return response.text


def convert_to_tfdata():
    # Connect to the Google Cloud Storage client
    bucket = connect_to_bucket()
    blobs = bucket.list_blobs(prefix=LABELED_DIR)

    for blob in blobs:
        # Load the JSON data from the blob
        blob_data = blob.download_as_text()
        try:
            json_data = json.loads(blob_data)
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from blob {blob.name}. Skipping this blob.")
            continue

        # Extract and preprocess data and labels from json_data
        data = [json_data['record']]
        label = [json_data['label']]

        # Convert data and labels into a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((data, label))

        # Define the destination blob name and upload
        dest_blob_name = f"{TF_DATA_DIR}/{blob.name.split('/')[-1].replace('.json', '')}"
        tf_data_path = f"/tmp/{blob.name.split('/')[-1].replace('.json', '')}"

        # Save the dataset in the TFRecord format
        tf.data.experimental.save(dataset, tf_data_path, compression='GZIP')

        for root, _, files in os.walk(tf_data_path):
            for file in files:
                local_file = os.path.join(root, file)
                blob = bucket.blob(f"{dest_blob_name}/{file}")
                blob.upload_from_filename(local_file)
                print(f"Uploaded {local_file} to {blob.name}")

        # Remove the local TFRecord directory if you want to clean up
        shutil.rmtree(tf_data_path)

        print(f"Uploaded tf.data data to {dest_blob_name}")


def prepare_data():
    # Connect to the Google Cloud Storage client
    bucket = connect_to_bucket()
    blobs = bucket.list_blobs(prefix=LABELED_DIR)
    all_data = []

    for blob in blobs:
        try:
            # Load the JSON data from the blob
            json_data = json.loads(blob.download_as_text())
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from blob {blob.name}. Skipping this blob.")
            continue
        else:
            # Extract input_text and output_text
            record = json.dumps(json_data["record"])
            label = json.dumps(json_data["label"])

            input_text = """
            Check if this email json includes one or more events and if so, identify the date, time and name of the event, if they exist. 
           Return the date, time and name in a new json format with the label 'label' and say nothing else. 
           If they do not exist, return the same json format with the date, time and name as null.
           """ + record

            # Format the data
            data_format = {"input_text": input_text, "output_text": label}
            all_data.append(data_format)

    # Split the data into train, test, and eval sets
    train_data, temp_data = train_test_split(
        all_data, test_size=0.3, random_state=42)
    test_data, eval_data = train_test_split(
        temp_data, test_size=0.5, random_state=42)

    # Save and upload the split data back to the bucket
    for split_name, split_data in zip(
        ["train", "test", "eval"], [train_data, test_data, eval_data]
    ):
        # Convert the data to JSONL format
        split_data_jsonl = "\n".join([json.dumps(item) for item in split_data])

        # Create a new blob in the bucket
        blob = bucket.blob(f"split/{split_name}.jsonl")

        # Upload the JSONL data to the blob
        blob.upload_from_string(
            split_data_jsonl, content_type="application/jsonl")


if __name__ == "__main__":
    # Generate the inputs arguments parser›
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(
        description="Generate and store training data")

    # Add command line options
    parser.add_argument(
        "cmd_name",
        choices=["setup", "generate", "tf_data", "prepare", "list", "show"],
        help="Action",
    )
    parser.add_argument(
        "-n", "--num_training", type=int, help="Number of training examples", default=1
    )
    parser.add_argument("-id", "--guid", type=str,
                        help="guid of a training example")
    args = parser.parse_args()
    print("Args:", args)

    if args.cmd_name == "setup":
        setup()
    if args.cmd_name == "generate":
        create_and_upload_generated_data(args.num_training)
    if args.cmd_name == "tf_data":
        convert_to_tfdata()
    if args.cmd_name == "prepare":
        prepare_data()
    if args.cmd_name == "list":
        list()
    if args.cmd_name == "show":
        if args.guid is not None:
            show(args.guid)
        else:
            show()
