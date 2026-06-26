"""UI Component for the AI Text-to-SQL Assistant."""

import streamlit as st

from src.platform.agentic_ai.sql_agent import (
    execute_agent_query,
    generate_sql,
    synthesize_results,
    validate_and_secure_sql,
    is_mock_llm,
)


def render_ai_chat():
    """Renders the AI Assistant chat interface."""
    st.subheader("💬 AI Database Assistant")
    st.caption("Ask questions in plain English to query the Data Warehouse.")

    if is_mock_llm():
        st.warning("⚠️ **Mock Mode Active:** Set `GEMINI_API_KEY` in a `.env` file (or environment variables) to connect to the live Gemini AI Agent.")


    # Custom CSS to make the chat container look distinct
    st.markdown(
        """
        <style>
        .ai-chat-container {
            border-left: 2px solid #FF5A5F;
            padding-left: 15px;
            height: 100%;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    query = st.chat_input("E.g., What are the top 5 most expensive neighbourhoods?")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            try:
                # 1. Show Agent Reasoning
                with st.status("Agent Reasoning...", expanded=True) as status:
                    st.write("🧠 Translating natural language to DuckDB SQL...")
                    raw_sql = generate_sql(query)
                    st.code(raw_sql, language="sql")

                    st.write("🛡️ Validating security guardrails & injecting LIMIT...")
                    safe_sql = validate_and_secure_sql(raw_sql)

                    st.write("⚡ Executing against analytical star schema...")
                    df = execute_agent_query(safe_sql)
                    status.update(
                        label="Query Execution Complete",
                        state="complete",
                        expanded=False,
                    )

                # 2. Show Data
                st.dataframe(df, width="stretch")

                # 3. Show Synthesis
                with st.spinner("Synthesizing executive summary..."):
                    summary = synthesize_results(query, df)
                st.markdown(summary)

            except ValueError as ve:
                st.error(str(ve))
            except RuntimeError as re:
                st.error(str(re))
                st.info(
                    "💡 Try rephrasing your question to match the schema properties (e.g., 'price_usd' instead of 'cost')."
                )
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
