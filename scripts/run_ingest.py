import argparse
import asyncio
from typing import Optional
from eaia.gmail import fetch_group_emails
from eaia.main.config import get_config
from langgraph_sdk import get_client
import httpx
import uuid
import hashlib
import lancedb
import os
import pyarrow as pa
from langchain_openai import OpenAIEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create embeddings directory if it doesn't exist
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

async def main(
    url: Optional[str] = None,
    minutes_since: int = 60,
    gmail_token: Optional[str] = None,
    gmail_secret: Optional[str] = None,
    early: bool = True,
    rerun: bool = False,
    email: Optional[str] = None,
):
    if email is None:
        email_address = get_config({"configurable": {}})["email"]
    else:
        email_address = email
    if url is None:
        client = get_client(url="http://127.0.0.1:2024")
    else:
        client = get_client(url=url)

    logger.info(f"Initializing LanceDB at {EMBEDDINGS_DIR}")
    # Initialize LanceDB
    db = lancedb.connect(EMBEDDINGS_DIR)
    
    # Define table schema using PyArrow
    schema = pa.schema([
        ('id', pa.string()),
        ('thread_id', pa.string()),
        ('from_email', pa.string()),
        ('to_email', pa.string()),
        ('subject', pa.string()),
        ('page_content', pa.string()),
        ('send_time', pa.string()),
        ('vector', pa.list_(pa.float32(), 1536))  # OpenAI embedding dimension
    ])
    
    logger.info("Creating/opening emails table")
    # Get or create the emails table
    if "emails" in db.table_names():
        table = db.open_table("emails")
        logger.info("Opened existing emails table")
    else:
        table = db.create_table("emails", schema=schema)
        logger.info("Created new emails table")

    # Initialize embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info("Initialized OpenAI embeddings model")

    email_count = 0
    for email in fetch_group_emails(
        email_address,
        minutes_since=minutes_since,
        gmail_token=gmail_token,
        gmail_secret=gmail_secret,
    ):
        logger.info(f"Processing email with ID: {email['id']}")
        thread_id = str(
            uuid.UUID(hex=hashlib.md5(email["thread_id"].encode("UTF-8")).hexdigest())
        )
        try:
            thread_info = await client.threads.get(thread_id)
        except httpx.HTTPStatusError as e:
            if "user_respond" in email:
                logger.info("Skipping user response email")
                continue
            if e.response.status_code == 404:
                thread_info = await client.threads.create(thread_id=thread_id)
                logger.info("Created new thread")
            else:
                logger.error(f"Error processing thread: {e}")
                raise e
        if "user_respond" in email:
            await client.threads.update_state(thread_id, None, as_node="__end__")
            logger.info("Updated thread state for user response")
            continue
        recent_email = thread_info["metadata"].get("email_id")
        if recent_email == email["id"]:
            if early:
                logger.info("Breaking early due to seen email")
                break
            else:
                if rerun:
                    logger.info("Rerunning seen email")
                    pass
                else:
                    logger.info("Skipping seen email")
                    continue

        try:
            # Generate embedding for the email content
            text_to_embed = f"Subject: {email['subject']}\nFrom: {email['from_email']}\nTo: {email.get('to_email', '')}\nContent: {email['page_content']}"
            vector = embeddings.embed_query(text_to_embed)
            logger.info("Generated embedding for email")
            
            # Save to LanceDB
            table.add([{
                "id": email["id"],
                "thread_id": thread_id,
                "from_email": email["from_email"],
                "to_email": email.get("to_email", ""),
                "subject": email["subject"],
                "page_content": email["page_content"],
                "send_time": email["send_time"],
                "vector": vector
            }])
            logger.info("Added email to LanceDB")
            email_count += 1
        except Exception as e:
            logger.error(f"Error saving email to LanceDB: {e}")
            raise e

        await client.threads.update(thread_id, metadata={"email_id": email["id"]})
        logger.info("Updated thread metadata")

        # Create a run and set the initial state to stop at human_node
        await client.runs.create(
            thread_id,
            "main",
            input={
                "email": email,
                "messages": [],  # Initialize empty messages
                "next": "human_node"  # This should make it stop at human interaction
            },
            multitask_strategy="rollback",
        )
        logger.info("Created new run that will stop for human review")

    logger.info(f"Processed {email_count} emails")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to run against",
    )
    parser.add_argument(
        "--early",
        type=int,
        default=1,
        help="whether to break when encountering seen emails",
    )
    parser.add_argument(
        "--rerun",
        type=int,
        default=0,
        help="whether to rerun all emails",
    )
    parser.add_argument(
        "--minutes-since",
        type=int,
        default=60,
        help="Only process emails that are less than this many minutes old.",
    )
    parser.add_argument(
        "--gmail-token",
        type=str,
        default=None,
        help="The token to use in communicating with the Gmail API.",
    )
    parser.add_argument(
        "--gmail-secret",
        type=str,
        default=None,
        help="The creds to use in communicating with the Gmail API.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="The email address to use",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            url=args.url,
            minutes_since=args.minutes_since,
            gmail_token=args.gmail_token,
            gmail_secret=args.gmail_secret,
            early=bool(args.early),
            rerun=bool(args.rerun),
            email=args.email,
        )
    )
