"""
Setup script to initialize the ticket analysis system.
Run this once to build the vector index.
"""

import os
from data_processor import TicketDataProcessor
from vector_store import TicketVectorStore


def main():
    """Build and save the vector index."""
    print("=" * 60)
    print("GenAI Ticket Analysis System - Setup")
    print("=" * 60)
    
    data_path = 'telecom_tickets_10000_12cats.json'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        return
    
    print(f"\n📂 Loading tickets from {data_path}...")
    processor = TicketDataProcessor(data_path)
    tickets = processor.load_tickets()
    
    print(f"✅ Loaded {len(tickets)} tickets")
    
    print("\n📝 Preparing ticket texts...")
    ticket_texts = processor.get_all_ticket_texts()
    ticket_metadata = processor.get_ticket_metadata()
    
    print("🔧 Building vector index (this may take a few minutes)...")
    vector_store = TicketVectorStore()
    vector_store.build_index(ticket_texts, ticket_metadata)
    
    print("\n💾 Saving index to disk...")
    vector_store.save_index()
    
    print("\n✅ Setup complete! You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

