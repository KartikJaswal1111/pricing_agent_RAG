import streamlit as st
import os
from dotenv import load_dotenv
from pricing_agent_rag import RAGPricingAgent

load_dotenv()

st.set_page_config(page_title="AI Strategic Pricing Agent (RAG)", layout="wide")


@st.cache_resource
def load_agent(api_key: str) -> RAGPricingAgent:
    return RAGPricingAgent(api_key=api_key)

st.title("🤖 Strategic Pricing Agent — RAG Enhanced")
st.markdown("""
This agent uses **Retrieval-Augmented Generation (RAG)** to ground every price
recommendation with historical pricing decisions, category benchmarks, and proven
strategies — reducing hallucination and improving accuracy over the CoT-only baseline.
""")

# ---------------------------------------------------------------------------
# Sidebar — configuration & inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", "")
    )

    st.divider()
    st.header("Product Data")

    product_name = st.text_input("Product Name", value="Puma Sneakers")
    category = st.selectbox(
        "Category",
        ["Footwear", "Electronics", "Clothing", "Luxury Goods", "Other"],
    )
    cost = st.number_input("Cost Price ($)", min_value=0.0, value=60.0, step=1.0)
    current = st.number_input(
        "Current Price ($) — optional", min_value=0.0, value=0.0, step=1.0
    )
    target_margin = st.slider("Target Margin (%)", 0, 100, 30)
    competitor = st.number_input(
        "Competitor Price ($) — optional", min_value=0.0, value=100.0, step=1.0
    )
    elasticity = st.selectbox(
        "Price Elasticity", ["High", "Medium-High", "Medium", "Low"]
    )

    st.divider()
    st.caption("Powered by Groq · LangChain · ChromaDB · Sentence Transformers")

# ---------------------------------------------------------------------------
# Main — run analysis
# ---------------------------------------------------------------------------
if st.button("Run RAG Analysis", type="primary"):
    if not api_key:
        st.error("Please provide a Groq API Key to continue.")
    else:
        try:
            with st.spinner("Building knowledge base & retrieving relevant examples…"):
                agent = load_agent(api_key)

            with st.spinner("Agent is reasoning with retrieved context…"):
                recommendation = agent.get_price_recommendation(
                    product_name=product_name,
                    category=category,
                    cost_price=cost,
                    current_price=current if current > 0 else None,
                    target_margin=target_margin,
                    competitor_price=competitor if competitor > 0 else None,
                    price_elasticity=elasticity,
                )

            st.subheader("RAG-Grounded Analysis & Recommendation")
            st.markdown(recommendation)

            # ---------------------------------------------------------------
            # Show retrieved context in an expander
            # ---------------------------------------------------------------
            with st.expander("View Retrieved Knowledge Base Context"):
                query = f"{category} {product_name} pricing {elasticity} elasticity"
                docs = agent.retrieve_relevant_knowledge(query, k=6)
                for i, doc in enumerate(docs, 1):
                    doc_type = doc.metadata.get("type", "unknown").replace("_", " ").title()
                    st.markdown(f"**Reference {i} — {doc_type}**")
                    st.code(doc.page_content, language=None)

        except Exception as e:
            st.error(f"An error occurred: {e}")
