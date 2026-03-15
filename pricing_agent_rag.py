"""
pricing_agent_rag.py: Core logic for the RAG-Enhanced Strategic Pricing Agent.
Uses a vector store of historical pricing decisions, category benchmarks,
and proven strategies to ground every recommendation with factual data.
"""

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os


# ---------------------------------------------------------------------------
# Pricing Knowledge Base Data
# ---------------------------------------------------------------------------

ELASTICITY_BENCHMARKS = [
    {
        "category": "Electronics",
        "elasticity": "High",
        "elasticity_coefficient": -2.1,
        "recommendation": "Small price changes (5-10%) to avoid demand drops",
    },
    {
        "category": "Footwear",
        "elasticity": "Medium-High",
        "elasticity_coefficient": -1.6,
        "recommendation": "Moderate price changes (10-15%) acceptable for premium brands",
    },
    {
        "category": "Clothing",
        "elasticity": "Medium",
        "elasticity_coefficient": -1.2,
        "recommendation": "Price changes up to 20% workable with good positioning",
    },
    {
        "category": "Luxury Goods",
        "elasticity": "Low",
        "elasticity_coefficient": -0.8,
        "recommendation": "Higher margins possible; price can signal quality",
    },
]

HISTORICAL_MARGINS = [
    {"category": "Electronics", "avg_margin": "12-18%", "range": "8-25%"},
    {"category": "Footwear",    "avg_margin": "25-35%", "range": "15-50%"},
    {"category": "Clothing",    "avg_margin": "30-45%", "range": "20-60%"},
    {"category": "Luxury Goods","avg_margin": "50-70%", "range": "40-80%"},
]

PRICING_DECISIONS = [
    {
        "product": "Samsung Galaxy Smartphone",
        "category": "Electronics",
        "cost": 450, "recommended_price": 499, "margin": "11%",
        "competitor_price": 529,
        "outcome": "Successful - gained 15% market share",
        "reasoning": "Aggressive pricing in high-elasticity category drove volume",
    },
    {
        "product": "Nike Running Shoes",
        "category": "Footwear",
        "cost": 80, "recommended_price": 140, "margin": "43%",
        "competitor_price": 150,
        "outcome": "Excellent - maintained premium positioning",
        "reasoning": "Brand strength allowed healthy margin while staying competitive",
    },
    {
        "product": "Adidas Sneakers",
        "category": "Footwear",
        "cost": 65, "recommended_price": 110, "margin": "41%",
        "competitor_price": 120,
        "outcome": "Good - achieved target volume",
        "reasoning": "Positioned below Nike but above discount brands",
    },
    {
        "product": "Puma Training Shoes",
        "category": "Footwear",
        "cost": 60, "recommended_price": 95, "margin": "37%",
        "competitor_price": 100,
        "outcome": "Solid - captured price-sensitive segment",
        "reasoning": "Value positioning with acceptable margin for volume growth",
    },
    {
        "product": "Designer Handbag",
        "category": "Luxury Goods",
        "cost": 200, "recommended_price": 650, "margin": "69%",
        "competitor_price": 700,
        "outcome": "Excellent - maintained exclusivity",
        "reasoning": "Luxury positioning where higher price reinforces prestige",
    },
    {
        "product": "H&M T-Shirt",
        "category": "Clothing",
        "cost": 8, "recommended_price": 15, "margin": "47%",
        "competitor_price": 18,
        "outcome": "Great - high volume sales",
        "reasoning": "Fast-fashion model with aggressive pricing for turnover",
    },
    {
        "product": "Laptop Computer",
        "category": "Electronics",
        "cost": 800, "recommended_price": 950, "margin": "16%",
        "competitor_price": 999,
        "outcome": "Good - competitive positioning",
        "reasoning": "Electronics require thin margins but volume compensates",
    },
]

PRICING_GUIDELINES = [
    {
        "rule": "High Elasticity Strategy",
        "guideline": "For high-elasticity products, keep price changes under 10% to avoid significant demand reduction",
    },
    {
        "rule": "Competitive Positioning",
        "guideline": "Stay within 15% of main competitor unless you have clear differentiation",
    },
    {
        "rule": "Margin Targets",
        "guideline": "Ensure minimum viable margin covers fixed costs - typically 15% for most categories",
    },
    {
        "rule": "Premium Positioning",
        "guideline": "For premium brands, price 10-30% above mass market to signal quality",
    },
]


# ---------------------------------------------------------------------------
# Document Builder
# ---------------------------------------------------------------------------

def _build_documents() -> List[Document]:
    """Convert all knowledge-base entries into LangChain Documents."""
    docs: List[Document] = []

    for b in ELASTICITY_BENCHMARKS:
        docs.append(Document(
            page_content=(
                f"Category: {b['category']}\n"
                f"Elasticity Level: {b['elasticity']}\n"
                f"Elasticity Coefficient: {b['elasticity_coefficient']}\n"
                f"Pricing Recommendation: {b['recommendation']}"
            ),
            metadata={"type": "elasticity_benchmark", "category": b["category"]},
        ))

    for m in HISTORICAL_MARGINS:
        docs.append(Document(
            page_content=(
                f"Category: {m['category']}\n"
                f"Average Margin: {m['avg_margin']}\n"
                f"Margin Range: {m['range']}"
            ),
            metadata={"type": "historical_margin", "category": m["category"]},
        ))

    for d in PRICING_DECISIONS:
        docs.append(Document(
            page_content=(
                f"Product: {d['product']}\n"
                f"Category: {d['category']}\n"
                f"Cost: ${d['cost']}\n"
                f"Recommended Price: ${d['recommended_price']}\n"
                f"Margin: {d['margin']}\n"
                f"Competitor Price: ${d['competitor_price']}\n"
                f"Outcome: {d['outcome']}\n"
                f"Reasoning: {d['reasoning']}"
            ),
            metadata={"type": "pricing_decision", "category": d["category"], "product": d["product"]},
        ))

    for g in PRICING_GUIDELINES:
        docs.append(Document(
            page_content=(
                f"Rule: {g['rule']}\n"
                f"Guideline: {g['guideline']}"
            ),
            metadata={"type": "pricing_guideline", "rule": g["rule"]},
        ))

    return docs


# ---------------------------------------------------------------------------
# RAG Pricing Agent
# ---------------------------------------------------------------------------

class RAGPricingAgent:
    """
    A RAG-enhanced pricing expert that grounds every recommendation
    with historical decisions, category benchmarks, and proven guidelines.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        persist_dir: str = "./pricing_knowledge_db",
    ):
        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024,
        )

        self.system_message = SystemMessage(content="""
            You are an expert pricing strategist with access to a comprehensive pricing knowledge base.

            Your capabilities:
            - Access to historical pricing decisions and their outcomes
            - Category-specific elasticity benchmarks and margin data
            - Proven pricing guidelines and best practices
            - Ability to ground recommendations with factual data

            Always:
            1. Use retrieved historical examples to support your recommendations
            2. Reference relevant category benchmarks and guidelines
            3. Explain how past similar decisions performed
            4. Provide data-backed reasoning for your price recommendation
            5. Address potential risks based on historical patterns
        """)

        # Build vector store (load from disk if it exists)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
        else:
            docs = _build_documents()
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=persist_dir,
            )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def retrieve_relevant_knowledge(self, query: str, k: int = 6) -> List[Document]:
        """Retrieve the k most relevant documents for a query."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            doc_type = doc.metadata.get("type", "unknown")
            parts.append(f"Reference {i} ({doc_type}):\n{doc.page_content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Core recommendation
    # ------------------------------------------------------------------

    def get_price_recommendation(
        self,
        product_name: str,
        category: str,
        cost_price: float,
        current_price: Optional[float] = None,
        target_margin: Optional[float] = None,
        competitor_price: Optional[float] = None,
        price_elasticity: Optional[str] = None,
    ) -> str:
        """Generate a RAG-grounded price recommendation."""

        # Build search query
        query = f"{category} {product_name} pricing elasticity margin cost {cost_price}"
        if competitor_price:
            query += f" competitor {competitor_price}"
        if price_elasticity:
            query += f" {price_elasticity} elasticity"

        # Retrieve & format context
        docs = self.retrieve_relevant_knowledge(query)
        context = self._format_context(docs)

        # Optional fields in prompt
        optional_lines = []
        if current_price is not None:
            optional_lines.append(f"Current Price: ${current_price}")
        if target_margin is not None:
            optional_lines.append(f"Target Margin: {target_margin}%")
        if competitor_price is not None:
            optional_lines.append(f"Competitor Price: ${competitor_price}")
        if price_elasticity is not None:
            optional_lines.append(f"Price Elasticity: {price_elasticity}")
        optional_block = "\n".join(optional_lines)

        prompt = f"""
PRICING REQUEST:
Product: {product_name}
Category: {category}
Cost Price: ${cost_price}
{optional_block}

RELEVANT HISTORICAL DATA AND BENCHMARKS:
{context}

Based on the above historical data and benchmarks, provide a comprehensive pricing
recommendation following this structure:

1. SIMILAR CASES: Reference 2-3 most relevant historical examples
2. CATEGORY ANALYSIS: Use category-specific elasticity and margin data
3. PRICE CALCULATION: Show your reasoning with referenced benchmarks
4. FINAL RECOMMENDATION: Specific price with confidence level
5. RISK ASSESSMENT: Based on similar historical outcomes

Analysis:
"""

        messages = [self.system_message, HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content
