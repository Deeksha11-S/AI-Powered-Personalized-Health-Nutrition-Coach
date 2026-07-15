import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime

#database initialization
nutrition_db = {
    "foods": {
        "chicken_breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
        "salmon": {"calories": 208, "protein": 20, "carbs": 0, "fat": 13},
        "brown_rice": {"calories": 216, "protein": 5, "carbs": 45, "fat": 1.8},
        "broccoli": {"calories": 55, "protein": 3.7, "carbs": 11, "fat": 0.6}
    },
    "meals": {
        "balanced_meal": ["chicken_breast", "brown_rice", "broccoli"],
        "high_protein": ["chicken_breast", "salmon", "broccoli"]
    }
}

exercise_db = {
    "cardio": {
        "running": {"calories": 600, "intensity": "high"},
        "cycling": {"calories": 400, "intensity": "medium"},
        "swimming": {"calories": 500, "intensity": "medium-high"}
    },
    "strength": {
        "weight_lifting": {"muscle_groups": "full_body", "intensity": "high"},
        "bodyweight": {"muscle_groups": "full_body", "intensity": "medium"},
        "resistance_bands": {"muscle_groups": "full_body", "intensity": "low-medium"}
    }
}

config = {
    "nutrition_goals": {
        "weight_loss": {"calorie_deficit": 500, "protein_ratio": 0.3},
        "muscle_gain": {"calorie_surplus": 300, "protein_ratio": 0.4},
        "maintenance": {"calorie_deficit": 0, "protein_ratio": 0.25}
    },
    "activity_levels": {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extremely_active": 1.9
    }
}

#core functions
def calculate_bmr(weight, height, age, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure"""
    activity_factor = config["activity_levels"].get(
        activity_level.lower(),
        1.2
    )
    return bmr * activity_factor

def calculate_macros(tdee, goal):
    """Calculate macronutrient distribution based on goals"""
    goal = goal.lower()
    goal_config = config["nutrition_goals"].get(
        goal,
        config["nutrition_goals"]["maintenance"]
    )

    if "calorie_deficit" in goal_config:
        calories = tdee - goal_config["calorie_deficit"]
    else:
        calories = tdee + goal_config["calorie_surplus"]

    protein_g = round((calories * goal_config["protein_ratio"]) / 4)
    fat_g = round((calories * 0.25) / 9)  # 25% of calories from fat
    carbs_g = round((calories - (protein_g * 4) - (fat_g * 9)) / 4)

    return {
        "calories": round(calories),
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g
    }

#classes
class UserProfile:
    def __init__(self):
        self.profiles = pd.DataFrame(columns=[
            'user_id', 'age', 'gender', 'weight', 'height',
            'goal', 'activity_level', 'bmr', 'tdee', 'cluster'
        ])
        self.next_id = 1
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.nn_model = NearestNeighbors(n_neighbors=5)

    def create_profile(self, age, gender, weight, height, goal, activity_level):
        """Create and store a new user profile"""
        try:
            if not all([age, gender, weight, height, goal, activity_level]):
                raise ValueError("All profile fields are required")

            bmr = calculate_bmr(weight, height, age, gender)
            tdee = calculate_tdee(bmr, activity_level)

            profile = {
                'user_id': self.next_id,
                'age': max(18, min(100, age)),
                'gender': gender.lower(),
                'weight': max(30, min(200, weight)),
                'height': max(120, min(250, height)),
                'goal': goal.lower(),
                'activity_level': activity_level.lower(),
                'bmr': bmr,
                'tdee': tdee,
                'cluster': -1
            }

            self.profiles = pd.concat([self.profiles, pd.DataFrame([profile])],
                                    ignore_index=True)
            self.next_id += 1
            self.update_clustering()

            return profile

        except Exception as e:
            print(f"Error creating profile: {str(e)}")
            raise

    def update_clustering(self):
        """Update user clusters when we have enough profiles"""
        if len(self.profiles) >= 5:
            features = self.profiles[['age', 'weight', 'height', 'bmr', 'tdee']]
            scaled_features = self.scaler.fit_transform(features)
            self.profiles['cluster'] = self.kmeans.fit_predict(scaled_features)
            self.nn_model.fit(scaled_features)

    def get_similar_users(self, user_id):
        """Find users with similar profiles"""
        if len(self.profiles) < 2 or user_id not in self.profiles['user_id'].values:
            return []

        user_idx = self.profiles.index[self.profiles['user_id'] == user_id].tolist()[0]
        user_data = self.scaler.transform([self.profiles.loc[user_idx, ['age', 'weight', 'height', 'bmr', 'tdee']]])

        distances, indices = self.nn_model.kneighbors(user_data)
        return self.profiles.iloc[indices[0]].to_dict('records')


class RecommendationEngine:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.feedback_db = pd.DataFrame(columns=[
            'user_id', 'date', 'weight', 'energy_level', 'adherence', 'feedback'
        ])

    def generate_diet_plan(self, user_id):
        """Generate personalized diet plan with error handling"""
        try:
            user = self.user_profile.profiles[
                self.user_profile.profiles['user_id'] == user_id
            ].iloc[0]

            macros = calculate_macros(user['tdee'], user['goal'])

            if user['goal'] == 'weight_loss':
                meal_plan = {
                    'breakfast': {"foods": ["oatmeal", "egg_whites"], "calories": 300},
                    'lunch': {"foods": ["chicken_breast", "brown_rice", "broccoli"], "calories": 500},
                    'dinner': {"foods": ["salmon", "quinoa", "asparagus"], "calories": 400},
                    'snacks': ["greek_yogurt", "almonds"]
                }
            elif user['goal'] == 'muscle_gain':
                meal_plan = {
                    'breakfast': {"foods": ["whole_eggs", "avocado", "toast"], "calories": 500},
                    'lunch': {"foods": ["lean_beef", "sweet_potato", "green_beans"], "calories": 600},
                    'dinner': {"foods": ["salmon", "quinoa", "broccoli"], "calories": 500},
                    'snacks': ["protein_shake", "cottage_cheese"]
                }
            else:
                meal_plan = {
                    'breakfast': {"foods": ["whole_grain_cereal", "milk", "berries"], "calories": 400},
                    'lunch': {"foods": ["turkey", "whole_wheat_bread", "salad"], "calories": 500},
                    'dinner': {"foods": ["chicken_thighs", "brown_rice", "mixed_veggies"], "calories": 500},
                    'snacks': ["fruit", "nuts"]
                }

            return {
                'macros': macros,
                'meal_plan': meal_plan
            }

        except Exception as e:
            print(f"Error generating diet plan: {str(e)}")
            return {
                'macros': {"calories": 2000, "protein_g": 150, "carbs_g": 200, "fat_g": 50},
                'meal_plan': {}
            }

    def generate_fitness_plan(self, user_id):
        """Generate personalized fitness plan with error handling"""
        try:
            user = self.user_profile.profiles[
                self.user_profile.profiles['user_id'] == user_id
            ].iloc[0]

            if user['goal'] == 'weight_loss':
                return {
                    'cardio': ["running", "cycling", "swimming"],
                    'strength': ["full_body_circuit", "bodyweight_exercises"],
                    'schedule': {
                        'Monday': "Cardio (30 min) + Full Body Strength",
                        'Wednesday': "HIIT (20 min)",
                        'Friday': "Cardio (45 min) + Core Work",
                        'Saturday': "Active Recovery (yoga/walking)"
                    }
                }
            elif user['goal'] == 'muscle_gain':
                return {
                    'strength': ["weight_lifting", "resistance_training"],
                    'cardio': ["light_jogging", "cycling"],
                    'schedule': {
                        'Monday': "Upper Body Strength",
                        'Tuesday': "Lower Body Strength",
                        'Thursday': "Upper Body Strength",
                        'Friday': "Lower Body Strength",
                        'Sunday': "Active Recovery"
                    }
                }
            else:
                return {
                    'strength': ["bodyweight_exercises", "resistance_bands"],
                    'cardio': ["brisk_walking", "cycling"],
                    'schedule': {
                        'Monday': "Full Body Strength",
                        'Wednesday': "Cardio (30 min)",
                        'Friday': "Full Body Strength",
                        'Sunday': "Active Recovery"
                    }
                }

        except Exception as e:
            print(f"Error generating fitness plan: {str(e)}")
            return {
                'cardio': [],
                'strength': [],
                'schedule': {}
            }

    def record_feedback(self, user_id, weight, energy_level, adherence, feedback):
        """Record user feedback with validation"""
        try:
            if user_id not in self.user_profile.profiles['user_id'].values:
                raise ValueError("Invalid user ID")

            new_entry = {
                'user_id': user_id,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'weight': max(30, min(200, weight)),
                'energy_level': max(1, min(10, energy_level)),
                'adherence': max(0, min(100, adherence)),
                'feedback': str(feedback)[:500]
            }

            self.feedback_db = pd.concat(
                [self.feedback_db, pd.DataFrame([new_entry])],
                ignore_index=True
            )

            print(f"Recorded feedback for user {user_id}")

            return True

        except Exception as e:
            print(f"Error recording feedback: {str(e)}")
            return False

#display functions
def format_profile(profile):
    """Format user profile for display with improved contrast"""
    return f"""
    <div style="font-family: Arial; padding: 20px; border-radius: 10px; background: #52c2c4; border: 1px solid #d1d5db;">
        <h2 style="color: #1e3a8a; margin-top: 0;">👤 Your Profile</h2>
        <p style="color: #1f2937;"><b>User ID:</b> {profile['user_id']} (save this for future use)</p>

        <div style="display: flex; gap: 15px; margin-top: 15px;">
            <div style="flex: 1; background: #52c2c4; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db;">
                <h3 style="color: #1d4ed8; margin-top: 0;">Basic Info</h3>
                <p style="color: #374151;">👶 <b>Age:</b> {profile['age']}</p>
                <p style="color: #374151;">🚻 <b>Gender:</b> {profile['gender'].capitalize()}</p>
                <p style="color: #374151;">⚖️ <b>Weight:</b> {profile['weight']} kg</p>
                <p style="color: #374151;">📏 <b>Height:</b> {profile['height']} cm</p>
            </div>

            <div style="flex: 1; background: #52c2c4; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db;">
                <h3 style="color: #1d4ed8; margin-top: 0;">Health Metrics</h3>
                <p style="color: #374151;">🔥 <b>BMR:</b> {round(profile['bmr'])} kcal/day</p>
                <p style="color: #374151;">🏃 <b>TDEE:</b> {round(profile['tdee'])} kcal/day</p>
                <p style="color: #374151;">🎯 <b>Goal:</b> {profile['goal'].capitalize()}</p>
                <p style="color: #374151;">🏋️ <b>Activity Level:</b> {profile['activity_level'].replace('_', ' ').title()}</p>
            </div>
        </div>
    </div>
    """

def format_diet_plan(plan):
    """Format diet plan for display with improved contrast"""
    macros = plan['macros']
    meals = plan['meal_plan']

    return f"""
    <div style="font-family: Arial; padding: 20px; border-radius: 10px; background: #52c2c4; border: 1px solid #d1d5db;">
        <h2 style="color: #1e3a8a; margin-top: 0;">🍽️ Your Personalized Diet Plan</h2>

        <div style="background: #52c2c4; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #d1d5db;">
            <h3 style="color: #1d4ed8; margin-top: 0;">📊 Daily Nutrition Targets</h3>
            <div style="display: flex; gap: 15px;">
                <div style="flex: 1;">
                    <p style="color: #374151;">🔥 <b>Calories:</b> {macros['calories']} kcal</p>
                    <p style="color: #374151;">🍗 <b>Protein:</b> {macros['protein_g']}g</p>
                </div>
                <div style="flex: 1;">
                    <p style="color: #374151;">🍞 <b>Carbs:</b> {macros['carbs_g']}g</p>
                    <p style="color: #374151;">🥑 <b>Fats:</b> {macros['fat_g']}g</p>
                </div>
            </div>
        </div>

        <div style="background: #52c2c4; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db;">
            <h3 style="color: #1d4ed8; margin-top: 0;">🍳 Sample Meal Plan</h3>

            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 5px; color: #1e40af;">🌅 Breakfast (~{meals['breakfast']['calories']} kcal)</h4>
                <p style="color: #374151;">{', '.join(meals['breakfast']['foods']).replace('_', ' ').title()}</p>
            </div>

            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 5px; color: #1e40af;">☀️ Lunch (~{meals['lunch']['calories']} kcal)</h4>
                <p style="color: #374151;">{', '.join(meals['lunch']['foods']).replace('_', ' ').title()}</p>
            </div>

            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 5px; color: #1e40af;">🌙 Dinner (~{meals['dinner']['calories']} kcal)</h4>
                <p style="color: #374151;">{', '.join(meals['dinner']['foods']).replace('_', ' ').title()}</p>
            </div>

            <div>
                <h4 style="margin-bottom: 5px; color: #1e40af;">🍏 Snacks</h4>
                <p style="color: #374151;">{', '.join(meals['snacks']).replace('_', ' ').title()}</p>
            </div>
        </div>
    </div>
    """

def format_fitness_plan(plan):
    """Format fitness plan for display with improved contrast"""
    return f"""
    <div style="font-family: Arial; padding: 20px; border-radius: 10px; background: #ffffff; border: 1px solid #d1d5db;">
        <h2 style="color: #1e3a8a; margin-top: 0;">💪 Your Personalized Fitness Plan</h2>

        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div style="flex: 1; background: #3c6bc9; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db;">
                <h3 style="color: #b91c1c; margin-top: 0;">🏃 Cardio Exercises</h3>
                <ul style="padding-left: 20px; color: #374151;">
                    {''.join([f'<li>{ex.capitalize()}</li>' for ex in plan['cardio']])}
                </ul>
            </div>

            <div style="flex: 1; background: #3c6bc9; padding: 15px; border-radius: 8px; border: 1px solid #d1d5db;">
                <h3 style="color: #b91c1c; margin-top: 0;">🏋️ Strength Exercises</h3>
                <ul style="padding-left: 20px; color: #374151;">
                    {''.join([f'<li>{ex.replace("_", " ").title()}</li>' for ex in plan['strength']])}
                </ul>
            </div>
        </div>
    </div>
    """

def create_macros_chart(macros):
    """Create pie chart of macronutrient distribution with improved contrast"""
    labels = ['Protein', 'Carbs', 'Fat']
    values = [macros['protein_g'], macros['carbs_g'], macros['fat_g']]

    fig = px.pie(
        names=labels,
        values=values,
        title="Macronutrient Distribution",
        color_discrete_sequence=['#1d4ed8', '#16a34a', '#d97706']  # Blue, Green, Amber
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont_color= '#7a2254')
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='#1672a3',
        plot_bgcolor='#16a389',
        font_color='#1f2937'
    )
    return fig

def create_fitness_chart(schedule):
    """Create bar chart of weekly schedule with improved contrast"""
    days = list(schedule.keys())
    activities = list(schedule.values())

    fig = px.bar(
        x=days,
        y=[1]*len(days),
        text=activities,
        title="Weekly Fitness Schedule",
        labels={'x': 'Day', 'y': ''},
        color=days,
        color_discrete_sequence=['#1d4ed8', '#9333ea', '#dc2626', '#16a34a', '#d97706', '#0284c7', '#c026d3']
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Day of Week",
        yaxis_visible=False,
        plot_bgcolor='#1672a3',
        paper_bgcolor='#16a389',
        font_color='#1f2937',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(textposition='inside', textfont_color='#a39516')
    return fig

def format_feedback(success):
    """Format feedback submission result with improved contrast"""
    if success:
        return """
        <div style="font-family: Arial; padding: 20px; border-radius: 10px; background: #7e60c4; border: 1px solid #16a34a;">
            <h2 style="color: #166534; margin-top: 0;">✅ Feedback Received!</h2>
            <p style="color: #1f2937;">Thank you for your feedback. We'll use this to improve your recommendations.</p>
            <p style="color: #1f2937;">Keep up the great work on your health journey! 💪</p>
        </div>
        """
    return """
    <div style="font-family: Arial; padding: 20px; border-radius: 10px; background: #7e60c4; border: 1px solid #dc2626;">
        <h2 style="color: #991b1b; margin-top: 0;">⚠️ Error Submitting Feedback</h2>
        <p style="color: #1f2937;">We couldn't process your feedback. Please try again later.</p>
    </div>
    """

def create_interface(user_system, recommendation_engine):
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as app:
        gr.Markdown("""
        <div style="text-align: center;">
            <h1 style="color: #2c3e50; margin-bottom: 10px;">🌟 Personalized Health & Nutrition Coach</h1>
            <p style="color: #7f8c8d;">Get customized diet and fitness plans tailored just for you!</p>
        </div>
        """)

        # Tab 1: Create Profile
        with gr.Tab("👤 Create Profile"):
            with gr.Row():
                with gr.Column():
                    age = gr.Number(label="Age", minimum=18, maximum=100, step=1)
                    gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
                    weight = gr.Number(label="Weight (kg)", minimum=30, maximum=200)
                    height = gr.Number(label="Height (cm)", minimum=120, maximum=250)
                with gr.Column():
                    goal = gr.Dropdown(
                        ["Weight Loss", "Muscle Gain", "Maintenance"],
                        label="Primary Goal"
                    )
                    activity_level = gr.Dropdown(
                        ["Sedentary", "Lightly Active", "Moderately Active",
                         "Very Active", "Extremely Active"],
                        label="Activity Level"
                    )
                    create_btn = gr.Button("Create My Profile", variant="primary")

            profile_output = gr.HTML(label="Your Profile Summary")

        # Tab 2: Get Recommendations
        with gr.Tab("📋 Get Recommendations"):
            with gr.Row():
                user_id = gr.Number(
                    label="Your User ID",
                    precision=0,
                    info="Enter the ID you received when creating your profile"
                )
                get_rec_btn = gr.Button("Get My Recommendations", variant="primary")

            with gr.Row():
                with gr.Column():
                    diet_output = gr.HTML(label="Your Diet Plan")
                    macros_chart = gr.Plot(label="Macronutrient Breakdown")
                with gr.Column():
                    fitness_output = gr.HTML(label="Your Fitness Plan")
                    fitness_chart = gr.Plot(label="Weekly Schedule")

        # Tab 3: Provide Feedback
        with gr.Tab("📝 Provide Feedback"):
            with gr.Row():
                with gr.Column():
                    fb_user_id = gr.Number(label="Your User ID", precision=0)
                    current_weight = gr.Number(label="Current Weight (kg)", minimum=30, maximum=200)
                    energy_level = gr.Slider(1, 10, label="Energy Level (1-10)", step=1)
                    adherence = gr.Slider(0, 100, label="Plan Adherence (%)", step=5)
                    feedback = gr.Textbox(
                        label="Your Feedback",
                        placeholder="How's it going? Any challenges?"
                    )
                    submit_fb = gr.Button("Submit Feedback", variant="primary")
                with gr.Column():
                    feedback_output = gr.HTML(label="Feedback Received")

        def handle_create_profile(age, gender, weight, height, goal, activity_level):
            try:
                if not all([age, gender, weight, height, goal, activity_level]):
                    return "<div style='color:red; padding:20px;'>Please fill in all fields</div>"

                profile = user_system.create_profile(age, gender, weight, height, goal, activity_level)
                return format_profile(profile)
            except Exception as e:
                return f"<div style='color:red; padding:20px;'>Error creating profile: {str(e)}</div>"

        def handle_get_recommendations(user_id):
            try:
                if not user_id or user_id not in user_system.profiles['user_id'].values:
                    return (
                        "<div style='color:red; padding:20px;'>Please enter a valid User ID</div>",
                        "<div style='color:red; padding:20px;'>Please enter a valid User ID</div>",
                        None,
                        None
                    )

                diet_plan = recommendation_engine.generate_diet_plan(user_id)
                fitness_plan = recommendation_engine.generate_fitness_plan(user_id)

                return (
                    format_diet_plan(diet_plan),
                    format_fitness_plan(fitness_plan),
                    create_macros_chart(diet_plan['macros']),
                    create_fitness_chart(fitness_plan['schedule'])
                )
            except Exception as e:
                error_msg = f"<div style='color:red; padding:20px;'>Error: {str(e)}</div>"
                return (error_msg, error_msg, None, None)

        def handle_submit_feedback(user_id, weight, energy_level, adherence, feedback):
            try:
                if not user_id or user_id not in user_system.profiles['user_id'].values:
                    return format_feedback(False)

                success = recommendation_engine.record_feedback(
                    user_id, weight, energy_level, adherence, feedback
                )
                return format_feedback(success)
            except Exception:
                return format_feedback(False)

        create_btn.click(
            fn=handle_create_profile,
            inputs=[age, gender, weight, height, goal, activity_level],
            outputs=profile_output
        )

        get_rec_btn.click(
            fn=handle_get_recommendations,
            inputs=user_id,
            outputs=[diet_output, fitness_output, macros_chart, fitness_chart]
        )

        submit_fb.click(
            fn=handle_submit_feedback,
            inputs=[fb_user_id, current_weight, energy_level, adherence, feedback],
            outputs=feedback_output
        )

    return app


if __name__ == "__main__":
    user_profile_system = UserProfile()
    recommendation_engine = RecommendationEngine(user_profile_system)

    interface = create_interface(user_profile_system, recommendation_engine)
    interface.launch(share=True)