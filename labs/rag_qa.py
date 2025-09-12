"""
RAG-based Q&A system for transaction data.
This provides immediate transaction Q&A without training.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re


class TransactionRAG:
    """Simple RAG system for answering questions about transaction data."""
    
    def __init__(self, csv_path: str = "data/all_transactions.csv"):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Amount_abs'] = self.df['Amount'].abs()
        
    def answer_question(self, question: str) -> str:
        """Answer a question about the transaction data."""
        question_lower = question.lower()
        
        # Total expenses
        if any(phrase in question_lower for phrase in ["total expense", "how much spent", "total spending"]):
            total_expenses = self.df[self.df['Type'] == 'Expense']['Amount'].abs().sum()
            return f"Your total expenses are ${total_expenses:,.2f}."
        
        # Total income
        if any(phrase in question_lower for phrase in ["total income", "how much earned", "total earning"]):
            total_income = self.df[self.df['Type'] == 'Income']['Amount'].sum()
            return f"Your total income is ${total_income:,.2f}."
            
        # Net income
        if any(phrase in question_lower for phrase in ["net income", "profit", "balance"]):
            total_income = self.df[self.df['Type'] == 'Income']['Amount'].sum()
            total_expenses = self.df[self.df['Type'] == 'Expense']['Amount'].abs().sum()
            net = total_income - total_expenses
            return f"Your net income is ${net:,.2f} (Income: ${total_income:,.2f} - Expenses: ${total_expenses:,.2f})."
        
        # Largest/biggest expense
        if any(phrase in question_lower for phrase in ["largest expense", "biggest expense", "most expensive", "highest expense"]):
            expense_data = self.df[self.df['Type'] == 'Expense']
            largest_expense = expense_data.loc[expense_data['Amount'].idxmin()]
            return f"Your largest expense was ${abs(largest_expense['Amount']):,.2f} for {largest_expense['Description']} on {largest_expense['Date'].strftime('%Y-%m-%d')}."
        
        # Recent transactions
        if any(phrase in question_lower for phrase in ["recent transaction", "latest transaction", "last transaction"]):
            recent = self.df.head(5)
            descriptions = []
            for _, row in recent.iterrows():
                descriptions.append(f"{row['Description']} (${abs(row['Amount']):,.2f}) on {row['Date'].strftime('%Y-%m-%d')}")
            return f"Your recent transactions are: {'; '.join(descriptions)}."
        
        # Category spending
        category_patterns = {
            'software': ['software', 'adobe', 'microsoft', 'office', 'subscription'],
            'computer_equipment': ['computer', 'equipment', 'hardware', 'tech'],
            'meals_entertainment': ['meal', 'food', 'entertainment', 'restaurant', 'lunch', 'dinner'],
            'fuel': ['fuel', 'gas', 'gasoline'],
            'office_supplies': ['office', 'supplies', 'paper', 'stationery'],
            'vehicle': ['vehicle', 'car', 'insurance', 'maintenance']
        }
        
        for category, keywords in category_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                category_expenses = self.df[
                    (self.df['Type'] == 'Expense') & 
                    (self.df['Category'] == category)
                ]['Amount'].abs().sum()
                
                if category_expenses > 0:
                    return f"You spent ${category_expenses:,.2f} on {category.replace('_', ' ')}."
                else:
                    # Try searching in descriptions
                    desc_matches = self.df[
                        (self.df['Type'] == 'Expense') & 
                        (self.df['Description'].str.contains('|'.join(keywords), case=False, na=False))
                    ]
                    if not desc_matches.empty:
                        total = desc_matches['Amount'].abs().sum()
                        return f"You spent ${total:,.2f} on items related to {category.replace('_', ' ')}."
        
        # Merchant-specific questions  
        merchants = ['microsoft', 'adobe', 'ikea', 'starbucks', 'netflix', 'amazon', 'google']
        for merchant in merchants:
            if merchant in question_lower:
                merchant_transactions = self.df[
                    self.df['Description'].str.contains(merchant, case=False, na=False)
                ]
                if not merchant_transactions.empty:
                    total = merchant_transactions['Amount'].abs().sum()
                    count = len(merchant_transactions)
                    return f"You have {count} transaction(s) with {merchant.title()} totaling ${total:,.2f}."
        
        # Time-based questions (months)
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        for i, month in enumerate(months, 1):
            if month in question_lower:
                month_expenses = self.df[
                    (self.df['Type'] == 'Expense') & 
                    (self.df['Date'].dt.month == i)
                ]['Amount'].abs().sum()
                if month_expenses > 0:
                    return f"You spent ${month_expenses:,.2f} in {month.title()}."
        
        # Financial summary
        if any(phrase in question_lower for phrase in ["summary", "overview", "breakdown"]):
            total_income = self.df[self.df['Type'] == 'Income']['Amount'].sum()
            total_expenses = self.df[self.df['Type'] == 'Expense']['Amount'].abs().sum()
            net_income = total_income - total_expenses
            transaction_count = len(self.df)
            expense_count = len(self.df[self.df['Type'] == 'Expense'])
            income_count = len(self.df[self.df['Type'] == 'Income'])
            
            return f"Financial Summary: Total Income: ${total_income:,.2f}, Total Expenses: ${total_expenses:,.2f}, Net Income: ${net_income:,.2f}. You have {transaction_count} total transactions ({income_count} income, {expense_count} expenses)."
        
        # Default response
        return f"I can answer questions about your transactions like: 'What are my total expenses?', 'What was my largest expense?', 'How much did I spend on software?', 'Give me a financial summary'. Could you rephrase your question?"


def main():
    """Test the RAG system."""
    rag = TransactionRAG()
    
    test_questions = [
        "What was my largest expense?",
        "What are my total expenses?",
        "How much did I spend on software?",
        "Give me a financial summary",
        "What are my recent transactions?",
        "How much did I spend in December?",
        "How much have I spent at Microsoft?"
    ]
    
    for question in test_questions:
        answer = rag.answer_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()