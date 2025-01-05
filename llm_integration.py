from pydantic import BaseModel, Field
import ast
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Define the chess move suggestion schema with proper typing for the move list
class ChessMoveSuggestion(BaseModel):
    moves: str = Field(
        ...,
        description="A list of 7 tuples representing possible chess moves. Each tuple contains the starting and ending position in standard chess notation."
    )

# Define the system prompt
System_Prompt = """
You are a highly skilled chess strategist and assistant. Your task is to analyze a given chessboard state represented by a FEN string and provide the best possible moves in the format of a list of tuples, where each tuple represents a move.

- Input: The chessboard state will be provided as a FEN string, which describes the position of all pieces.
  Example:
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  In this notation:
    - Uppercase letters represent white pieces (e.g., R for Rook, N for Knight, B for Bishop, Q for Queen, K for King, P for Pawn).
    - Lowercase letters represent black pieces (e.g., r for Rook, n for Knight, b for Bishop, q for Queen, k for King, p for Pawn).
    - Numbers represent empty squares.
    - The next move (w or b) indicates whose turn it is.
    - Castling rights, en passant, half-move clock, and full move number follow in the FEN string.

- Output: Provide a JSON-compatible Python object in the format of:
  ```python
  "moves": [("starting_square", "ending_square"), ...]

- Limit the number of moves to 7.
  """
  
# Set up the GITHUB_TOKEN  and chat model
# Go to your Github profile --> Settings --> Developer settings --> Personal access tokens --> Tokens Classic --> Generate new token
# Github Marketplace  : https://github.com/marketplace/models/catalog


token = os.getenv("GITHUB_TOKEN")

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=token,
    base_url="https://models.inference.ai.azure.com",
)

LLM_Structured = LLM.with_structured_output(ChessMoveSuggestion)

prompt = ChatPromptTemplate.from_messages(
    [("system", System_Prompt), 
     ("human", "The actual Chessboard FEN is: \n\n {Actual_FEN}.")]
)

LLM_suggester = prompt | LLM_Structured

# Define the function to run the suggester
def Run_suggester(fen_string):
    Input = {"Actual_FEN": fen_string} 
    response = LLM_suggester.invoke(Input) 
    response = ast.literal_eval(response.moves)
    return response

# Test with an example FEN string
fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
Moves = Run_suggester(fen_string)
print(Moves)
print(len(Moves))
