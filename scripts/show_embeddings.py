import os
import lancedb
from tabulate import tabulate

# Get the embeddings directory path
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")

def main():
    # Connect to LanceDB
    db = lancedb.connect(EMBEDDINGS_DIR)
    
    # Check if emails table exists
    if "emails" not in db.table_names():
        print("No email embeddings found. Please run the ingest script first.")
        return
        
    # Open the emails table
    table = db.open_table("emails")
    
    # Get all records using to_arrow()
    arrow_table = table.to_arrow()
    
    # Convert to list of dictionaries for display
    display_records = []
    for i in range(len(arrow_table)):
        record = {
            'id': arrow_table['id'][i].as_py(),
            'thread_id': arrow_table['thread_id'][i].as_py(),
            'from_email': arrow_table['from_email'][i].as_py(),
            'to_email': arrow_table['to_email'][i].as_py(),
            'subject': arrow_table['subject'][i].as_py(),
            'send_time': arrow_table['send_time'][i].as_py(),
            'page_content': arrow_table['page_content'][i].as_py()[:100] + '...' 
                if len(arrow_table['page_content'][i].as_py()) > 100 
                else arrow_table['page_content'][i].as_py()
        }
        display_records.append(record)
    
    # Print table info
    print(f"\nTotal emails stored: {len(display_records)}")
    print("\nEmail Records:")
    print(tabulate(display_records, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    main() 