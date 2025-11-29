"""
Streamlit application for GenAI-powered ticket analysis.
"""

import streamlit as st
import os
from data_processor import TicketDataProcessor
from vector_store import TicketVectorStore
from solution_generator import SolutionGenerator


# Page configuration
st.set_page_config(
    page_title="GenAI Ticket Analysis Assistant",
    page_icon="🎫",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


@st.cache_resource
def initialize_system():
    """Initialize the ticket analysis system."""
    data_path = 'telecom_tickets_10000_12cats.json'
    
    # Load and process data
    processor = TicketDataProcessor(data_path)
    tickets = processor.load_tickets()
    ticket_texts = processor.get_all_ticket_texts()
    ticket_metadata = processor.get_ticket_metadata()
    
    # Initialize vector store
    vector_store = TicketVectorStore()
    
    # Try to load existing index, otherwise build new one
    if not vector_store.load_index():
        st.info("Building vector index (this may take a few minutes)...")
        vector_store.build_index(ticket_texts, ticket_metadata)
        vector_store.save_index()
    
    return vector_store, processor


def main():
    """Main application function."""
    st.title("🎫 GenAI-Powered Ticket Analysis Assistant")
    st.markdown("### Analyze telecom support tickets and get AI-suggested solutions")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            ["groq", "openai"],
            help="Choose your LLM provider. Groq offers fast inference, OpenAI offers high quality."
        )
        
        # Model selection based on provider
        if provider == "groq":
            model = st.selectbox(
                "Groq Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                help="llama-3.3-70b-versatile is recommended (replaces deprecated llama-3.1-70b-versatile)",
                index=0
            )
        else:
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="gpt-3.5-turbo is cost-effective, gpt-4 offers higher quality",
                index=0
            )
        
        # Store model in session state
        st.session_state.llm_model = model
        
        # API Key input based on provider
        if provider == "groq":
            # Groq key is hardcoded, no input needed
            api_key = os.getenv('GROQ_API_KEY') or "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"
            st.info("✅ Groq API key configured (hardcoded)")
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key. You can also set OPENAI_API_KEY environment variable.",
                value=os.getenv('OPENAI_API_KEY', '')
            )
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
        
        # Store provider and model in session state
        st.session_state.provider = provider
        if 'llm_model' not in st.session_state:
            st.session_state.llm_model = model
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Initialize system
        if st.button("🔄 Initialize System", type="primary"):
            with st.spinner("Initializing system..."):
                try:
                    vector_store, processor = initialize_system()
                    st.session_state.vector_store = vector_store
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing system: {str(e)}")
        
        if st.session_state.initialized:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System not initialized")
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please initialize the system from the sidebar first.")
        return
    
    if not api_key and provider != "groq":
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to generate solutions.")
    
    # Ticket input section
    st.header("📝 Enter Ticket Description")
    
    ticket_description = st.text_area(
        "Customer Issue Description",
        height=150,
        placeholder="Enter the customer's issue description here...\n\nExample: Customer reports intermittent internet connectivity on mobile data in the late afternoon."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        # Groq doesn't need API key check (hardcoded), OpenAI does
        button_disabled = False if provider == "groq" else not api_key
        analyze_button = st.button("🔍 Analyze Ticket", type="primary", disabled=button_disabled)
    
    # Analysis results
    if analyze_button and ticket_description:
        if not st.session_state.vector_store:
            st.error("System not initialized. Please initialize from the sidebar.")
            return
        
        with st.spinner("🔍 Analyzing ticket and generating solutions..."):
            try:
                # Search for similar tickets
                vector_store = st.session_state.vector_store
                # Retrieve more tickets for better diversity
                similar_tickets = vector_store.search(ticket_description, k=15)
                
                # Generate solutions
                if api_key:
                    # Use model from session state (set in sidebar)
                    selected_model = st.session_state.get('llm_model', model)
                    generator = SolutionGenerator(provider=provider, api_key=api_key, model=selected_model)
                    solutions = generator.generate_solutions(ticket_description, similar_tickets)
                else:
                    st.error("API key required for solution generation.")
                    return
                
                # Display results
                st.success("✅ Analysis complete!")
                
                # Show similar tickets
                with st.expander("📋 Similar Resolved Tickets (Reference)", expanded=False):
                    for i, (ticket, score) in enumerate(similar_tickets[:3], 1):
                        st.markdown(f"**Similar Ticket {i}** (Similarity: {score:.2%})")
                        st.markdown(f"- **Category:** {ticket.get('category', 'N/A')}")
                        st.markdown(f"- **Issue:** {ticket.get('issue', 'N/A')}")
                        st.markdown(f"- **Root Cause:** {ticket.get('root_cause', 'N/A')}")
                        st.markdown(f"- **Resolution:** {ticket.get('resolution', 'N/A')}")
                        st.markdown("---")
                
                # Display solutions
                st.header("💡 Recommended Solutions")
                st.markdown("### Top 3 Solutions Ranked by Suitability")
                
                for solution in solutions:
                    rank = solution.get('rank', 0)
                    suitability = solution.get('suitability_percentage', 0)
                    sol_text = solution.get('solution', 'No solution provided')
                    reasoning = solution.get('reasoning', 'Based on similar ticket analysis')
                    
                    # Color code by suitability
                    if suitability >= 80:
                        color = "🟢"
                    elif suitability >= 60:
                        color = "🟡"
                    else:
                        color = "🟠"
                    
                    with st.container():
                        st.markdown(f"### {color} Solution #{rank} - {suitability}% Suitability")
                        st.progress(suitability / 100)
                        st.markdown(f"**Solution:** {sol_text}")
                        st.markdown(f"**Reasoning:** {reasoning}")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>GenAI-Powered Ticket Analysis System | Built with Streamlit, FAISS, and OpenAI</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

