"""
Data preprocessing module for telecom tickets.
Handles loading, chunking, and preparing ticket data for embedding.
"""

import json
import os
from typing import List, Dict, Any


class TicketDataProcessor:
    """Processes telecom ticket data for embedding and retrieval."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the JSON file containing tickets
        """
        self.data_path = data_path
        self.tickets = []
    
    def load_tickets(self) -> List[Dict[str, Any]]:
        """
        Load tickets from JSON file.
        
        Returns:
            List of ticket dictionaries
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)
        
        print(f"Loaded {len(self.tickets)} tickets from {self.data_path}")
        return self.tickets
    
    def prepare_ticket_text(self, ticket: Dict[str, Any]) -> str:
        """
        Combine ticket fields into a searchable text string.
        
        Args:
            ticket: Ticket dictionary
            
        Returns:
            Combined text string
        """
        issue = ticket.get('customer_issue_description', '')
        root_cause = ticket.get('root_cause', '')
        resolution = ticket.get('final_resolution', '')
        category = ticket.get('category', '')
        
        # Combine all relevant information for better matching
        text = f"Category: {category}\n"
        text += f"Issue: {issue}\n"
        text += f"Root Cause: {root_cause}\n"
        text += f"Resolution: {resolution}"
        
        return text
    
    def get_all_ticket_texts(self) -> List[str]:
        """
        Get all ticket texts for embedding.
        
        Returns:
            List of combined ticket text strings
        """
        if not self.tickets:
            self.load_tickets()
        
        return [self.prepare_ticket_text(ticket) for ticket in self.tickets]
    
    def get_ticket_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for each ticket (for retrieval context).
        
        Returns:
            List of ticket metadata dictionaries
        """
        if not self.tickets:
            self.load_tickets()
        
        return [
            {
                'ticket_id': ticket.get('ticket_id'),
                'category': ticket.get('category'),
                'issue': ticket.get('customer_issue_description'),
                'root_cause': ticket.get('root_cause'),
                'resolution': ticket.get('final_resolution')
            }
            for ticket in self.tickets
        ]

