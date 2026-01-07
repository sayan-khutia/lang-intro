"""
Fitness Coach Agent with streaming support for API integration.
Uses LangChain/LangGraph with MongoDB for long-term memory.
"""

from dotenv import load_dotenv
import os
from datetime import datetime
import json
import asyncio
from typing import AsyncGenerator

load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.store.base import BaseStore
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient


# Initialize the model
model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.7
)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "fitness_coach_db")
STORE_COLLECTION_NAME = "fitness_memories"

# Create MongoDB client, checkpointer, and store
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]

# Checkpointer for short-term memory (session state)
checkpointer = MongoDBSaver(mongo_client, db_name=DB_NAME)

# Store for long-term memory - pass the collection directly
store_collection = db[STORE_COLLECTION_NAME]
memory_store = MongoDBStore(store_collection)


# Define fitness coach tools
def safe_int(value, field_name: str) -> int | str:
    """Safely convert value to int, return error message if invalid."""
    try:
        return int(float(value))  # Handle "30.0" -> 30
    except (ValueError, TypeError):
        return f"Error: {field_name} must be a number, got '{value}'"


def safe_float(value, field_name: str) -> float | str:
    """Safely convert value to float, return error message if invalid."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return f"Error: {field_name} must be a number, got '{value}'"


@tool
def log_workout(
    user_id: str,
    exercise_type: str,
    duration_minutes: str | int,
    calories_burned: str | int,
    notes: str = ""
) -> str:
    """Log a workout session for the user.

    Args:
        user_id: The user's unique identifier
        exercise_type: Type of exercise (e.g., 'running', 'weight training', 'yoga')
        duration_minutes: Duration of the workout in minutes (must be a number like 30 or 45)
        calories_burned: Estimated calories burned (must be a number like 200 or 350)
        notes: Additional notes about the workout
    """
    duration = safe_int(duration_minutes, "duration_minutes")
    if isinstance(duration, str):
        return duration
    calories = safe_int(calories_burned, "calories_burned")
    if isinstance(calories, str):
        return calories
    
    workout_data = {
        "exercise_type": exercise_type,
        "duration_minutes": duration,
        "calories_burned": calories,
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }
    return f"Workout logged successfully: {exercise_type} for {duration} minutes, {calories} calories burned."


@tool
def set_fitness_goal(
    user_id: str,
    goal_type: str,
    target_value: str | float,
    target_date: str,
    description: str
) -> str:
    """Set a fitness goal for the user.

    Args:
        user_id: The user's unique identifier
        goal_type: Type of goal (e.g., 'weight_loss', 'muscle_gain', 'endurance', 'flexibility')
        target_value: Target numeric value for the goal (e.g., 70 for weight in kg, 5 for distance in km)
        target_date: Target date to achieve the goal (YYYY-MM-DD format)
        description: Detailed description of the goal
    """
    target = safe_float(target_value, "target_value")
    if isinstance(target, str):
        return target
    return f"Fitness goal set: {goal_type} - Target: {target} by {target_date}. {description}"


@tool
def calculate_bmi(weight_kg: str | float, height_cm: str | float) -> str:
    """Calculate BMI (Body Mass Index) for the user.

    Args:
        weight_kg: Weight in kilograms (must be a number like 70 or 85.5)
        height_cm: Height in centimeters (must be a number like 170 or 175)
    """
    weight = safe_float(weight_kg, "weight_kg")
    if isinstance(weight, str):
        return weight
    height = safe_float(height_cm, "height_cm")
    if isinstance(height, str):
        return height
    
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return f"BMI: {bmi:.1f} - Category: {category}"


@tool
def get_exercise_recommendation(
    fitness_level: str,
    goal: str,
    available_time_minutes: str | int
) -> str:
    """Get personalized exercise recommendations.

    Args:
        fitness_level: Current fitness level ('beginner', 'intermediate', 'advanced')
        goal: Fitness goal ('weight_loss', 'muscle_gain', 'endurance', 'flexibility', 'general_fitness')
        available_time_minutes: Available time for workout in minutes (must be a number like 30 or 60)
    """
    time_mins = safe_int(available_time_minutes, "available_time_minutes")
    if isinstance(time_mins, str):
        return time_mins
    
    recommendations = {
        ("beginner", "weight_loss"): "Start with 20-30 min brisk walking, light jogging, or swimming. Add bodyweight exercises like squats and lunges.",
        ("beginner", "muscle_gain"): "Begin with bodyweight exercises: push-ups, squats, planks. Focus on form over intensity.",
        ("beginner", "endurance"): "Start with interval training: 1 min jog, 2 min walk. Gradually increase jogging duration.",
        ("intermediate", "weight_loss"): "HIIT workouts combining cardio bursts with strength training. Circuit training works great.",
        ("intermediate", "muscle_gain"): "Split routine with compound movements: bench press, deadlifts, rows. Progressive overload is key.",
        ("intermediate", "endurance"): "Tempo runs, cycling intervals, and swimming sets. Aim for 45-60 min sessions.",
        ("advanced", "weight_loss"): "High-intensity metabolic conditioning, complex movements, and sport-specific training.",
        ("advanced", "muscle_gain"): "Advanced split routines with periodization. Include drop sets and supersets.",
        ("advanced", "endurance"): "Long-duration training with race-pace intervals. Cross-training for recovery.",
    }
    
    key = (fitness_level.lower(), goal.lower())
    base_recommendation = recommendations.get(key, "Focus on a balanced mix of cardio and strength training.")
    
    if time_mins < 30:
        time_advice = "For short sessions, focus on compound movements and high intensity."
    elif time_mins < 60:
        time_advice = "Good duration for a complete workout with warm-up and cool-down."
    else:
        time_advice = "Great time available! Include thorough warm-up, main workout, and stretching."
    
    return f"{base_recommendation}\n\nTime advice: {time_advice}"


@tool
def calculate_daily_calories(
    weight_kg: str | float,
    height_cm: str | float,
    age: str | int,
    gender: str,
    activity_level: str,
    goal: str
) -> str:
    """Calculate recommended daily calorie intake.

    Args:
        weight_kg: Weight in kilograms (must be a number like 70 or 85.5)
        height_cm: Height in centimeters (must be a number like 170 or 175)
        age: Age in years (must be a number like 25 or 30)
        gender: Gender ('male' or 'female')
        activity_level: Activity level ('sedentary', 'light', 'moderate', 'active', 'very_active')
        goal: Goal ('maintain', 'lose', 'gain')
    """
    weight = safe_float(weight_kg, "weight_kg")
    if isinstance(weight, str):
        return weight
    height = safe_float(height_cm, "height_cm")
    if isinstance(height, str):
        return height
    user_age = safe_int(age, "age")
    if isinstance(user_age, str):
        return user_age
    
    # Calculate BMR using Mifflin-St Jeor equation
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * user_age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * user_age - 161
    
    # Activity multipliers
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    
    tdee = bmr * activity_multipliers.get(activity_level.lower(), 1.55)
    
    # Adjust for goal
    if goal.lower() == "lose":
        target_calories = tdee - 500  # 0.5 kg/week loss
        goal_note = "This creates a 500 calorie deficit for gradual weight loss."
    elif goal.lower() == "gain":
        target_calories = tdee + 300  # Lean bulk
        goal_note = "This creates a 300 calorie surplus for lean muscle gain."
    else:
        target_calories = tdee
        goal_note = "This maintains your current weight."
    
    return f"BMR: {bmr:.0f} cal | TDEE: {tdee:.0f} cal | Recommended daily intake: {target_calories:.0f} cal\n{goal_note}"


# Bind tools to model
tools = [log_workout, set_fitness_goal, calculate_bmi, get_exercise_recommendation, calculate_daily_calories]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# Fitness coach system prompt
SYSTEM_PROMPT = """You are an expert fitness coach with deep knowledge in:
- Exercise science and workout programming
- Nutrition and diet planning
- Weight management and body composition
- Sports performance and athletic training
- Injury prevention and recovery

Your personality:
- Motivating and supportive, but also realistic
- You celebrate progress and encourage consistency
- You personalize advice based on the user's history and goals
- You explain the 'why' behind your recommendations

When interacting with users:
1. Always refer to their past conversations and progress when available
2. Be specific with recommendations (sets, reps, duration, etc.)
3. Ask clarifying questions when needed
4. Provide actionable advice they can implement immediately
5. Track their journey and adjust recommendations based on progress

Use the available tools to:
- Log workouts and track progress
- Set and monitor fitness goals
- Calculate BMI and calorie needs
- Provide exercise recommendations

Remember previous conversations with the user to provide personalized, continuous coaching."""


def format_chat_history(history: list) -> str:
    """Format chat history for context."""
    if not history:
        return "No previous conversations found."
    
    formatted = []
    for entry in history[-10:]:  # Last 10 interactions
        timestamp = entry.get("timestamp", "Unknown time")
        summary = entry.get("summary", "")
        formatted.append(f"[{timestamp}] {summary}")
    
    return "\n".join(formatted)


@task
def call_fitness_llm(messages: list[BaseMessage], chat_history_context: str):
    """LLM decides whether to call a tool or provide advice."""
    system_with_context = f"""{SYSTEM_PROMPT}

Previous conversation history with this user:
{chat_history_context}
"""
    return model_with_tools.invoke(
        [SystemMessage(content=system_with_context)] + messages
    )


@task
def call_fitness_tool(tool_call: dict):
    """Execute the fitness tool."""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)


@entrypoint(checkpointer=checkpointer, store=memory_store)
def fitness_coach_agent(
    messages: list[BaseMessage],
    *,
    store: BaseStore,
    previous: list[BaseMessage] | None = None
) -> list[BaseMessage]:
    """
    Fitness Coach Agent with long-term memory.
    
    Uses MongoDB store for persistent memory across sessions.
    The 'previous' parameter maintains conversation state within a session.
    The 'store' parameter provides access to long-term memory across sessions.
    """
    # Get user_id from config (passed via configurable)
    from langgraph.config import get_config
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    # Namespace for user-specific memory
    namespace = ("fitness_coach", "users", user_id)
    
    # Retrieve past chat history from long-term memory
    chat_history_items = store.search(namespace)
    chat_history = [item.value for item in chat_history_items] if chat_history_items else []
    chat_history_context = format_chat_history(chat_history)
    
    # Combine previous messages with new ones
    if previous:
        messages = add_messages(previous, messages)
    
    # Call the LLM
    model_response = call_fitness_llm(messages, chat_history_context).result()
    
    # Tool calling loop
    while True:
        if not model_response.tool_calls:
            break
        
        # Execute tools
        tool_result_futures = [
            call_fitness_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_fitness_llm(messages, chat_history_context).result()
    
    # Add final response to messages
    messages = add_messages(messages, model_response)
    
    # Save conversation summary to long-term memory
    # Extract the latest user message and AI response for summary
    user_message = ""
    ai_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not ai_response:
            ai_response = msg.content[:200] if msg.content else ""
        elif isinstance(msg, HumanMessage) and not user_message:
            user_message = msg.content[:100] if msg.content else ""
        if user_message and ai_response:
            break
    
    # Store conversation summary in long-term memory
    conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    store.put(
        namespace,
        conversation_id,
        {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": ai_response,
            "summary": f"User: {user_message[:50]}... | Coach: {ai_response[:50]}..."
        }
    )
    
    return messages


# ============== API-Compatible Functions ==============

def chat_with_coach(user_id: str, message: str) -> str:
    """
    Chat with the fitness coach (non-streaming).
    
    Args:
        user_id: Unique identifier for the user
        message: User's message
    
    Returns:
        The coach's response as a string
    """
    config = {
        "configurable": {
            "thread_id": f"fitness_{user_id}",
            "user_id": user_id
        }
    }
    
    messages = [HumanMessage(content=message)]
    
    # Run the agent
    result = fitness_coach_agent.invoke(
        messages,
        config=config
    )
    
    # Return the last AI message
    for msg in reversed(result):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "I'm sorry, I couldn't process your request."


async def stream_chat_with_coach(user_id: str, message: str) -> AsyncGenerator[str, None]:
    """
    Chat with the fitness coach and return the complete response.
    
    Args:
        user_id: Unique identifier for the user
        message: User's message
    
    Yields:
        The complete response as a single chunk
    """
    config = {
        "configurable": {
            "thread_id": f"fitness_{user_id}",
            "user_id": user_id
        }
    }
    
    messages = [HumanMessage(content=message)]
    
    # Get user_id from config namespace
    namespace = ("fitness_coach", "users", user_id)
    
    # Retrieve past chat history from long-term memory
    chat_history_items = memory_store.search(namespace)
    chat_history = [item.value for item in chat_history_items] if chat_history_items else []
    chat_history_context = format_chat_history(chat_history)
    
    # Build system message with context
    system_with_context = f"""{SYSTEM_PROMPT}

Previous conversation history with this user:
{chat_history_context}
"""
    
    full_messages = [SystemMessage(content=system_with_context)] + messages
    full_response_content = ""
    
    # Tool calling loop - collect complete response before sending
    while True:
        # Collect the complete model response
        current_response_content = ""
        tool_calls = []
        
        async for chunk in model_with_tools.astream(full_messages):
            # Collect tool calls if present
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            
            # Collect content (don't yield yet)
            if chunk.content:
                current_response_content += chunk.content
        
        # If there are tool calls, execute them
        if tool_calls:
            # Create AI message with tool calls
            ai_message = AIMessage(content=current_response_content, tool_calls=tool_calls)
            full_messages.append(ai_message)
            
            # Execute all tools
            for tool_call in tool_calls:
                tool = tools_by_name.get(tool_call["name"])
                if tool:
                    result = tool.invoke(tool_call)
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                    full_messages.append(tool_message)
            
            # Continue the loop to get the final response
            continue
        else:
            # No tool calls, we have the final response
            full_response_content = current_response_content
            break
    
    # Yield the complete response as a single chunk
    yield full_response_content
    
    # Save conversation summary to long-term memory
    conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory_store.put(
        namespace,
        conversation_id,
        {
            "timestamp": datetime.now().isoformat(),
            "user_message": message[:100],
            "ai_response": full_response_content[:200] if full_response_content else "",
            "summary": f"User: {message[:50]}... | Coach: {full_response_content[:50] if full_response_content else ''}..."
        }
    )


def get_user_chat_history(user_id: str, limit: int = 10) -> list[dict]:
    """
    Get chat history from MongoDB memory store for a user.
    
    Args:
        user_id: Unique identifier for the user
        limit: Maximum number of history items to return
    
    Returns:
        List of chat history items
    """
    namespace = ("fitness_coach", "users", user_id)
    
    try:
        chat_history_items = memory_store.search(namespace)
        history = [item.value for item in chat_history_items] if chat_history_items else []
        
        # Sort by timestamp and limit
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history[:limit]
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []


# ============== Interactive CLI (for testing) ==============

def main():
    """Main function to run the fitness coach chatbot interactively."""
    print("=" * 60)
    print("🏋️  FITNESS COACH - Your Personal AI Trainer 🏋️")
    print("=" * 60)
    print("\nI'm your AI fitness coach with memory! I remember our")
    print("conversations and track your fitness journey over time.")
    print("\nType 'quit' to exit.\n")
    
    # Get user ID
    user_id = input("Enter your user ID (or press Enter for 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"
    
    print(f"\nWelcome, {user_id}! Let's work on your fitness goals.\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n💪 Keep pushing towards your goals! See you next time!")
                break
            
            if not user_input:
                continue
            
            print("\nCoach: ", end="", flush=True)
            response = chat_with_coach(user_id, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n💪 Keep pushing towards your goals! See you next time!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
