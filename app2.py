import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import altair as alt
from io import StringIO

# --- Configuration and Setup ---
st.set_page_config(
    page_title="Student Behaviour Tracker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for the dataframe and chat history
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Helper Functions ---

# Define the category columns for scoring
CATEGORY_COLS = [
    'Behaviour', 'Home Life', 'Eating Habits', 'Disabilities',
    'Interventions', 'Safeguarding', 'Social'
]

def process_data(df_raw):
    """Calculates Total Score, Risk Category, Sentiment, and Anomalies."""
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # 1. Calculate Total Score
    df['Total Score'] = df[CATEGORY_COLS].sum(axis=1)

    # 2. Calculate Risk Category
    def get_risk_category(score):
        if 0 <= score <= 24:
            return 'High Risk'
        elif 25 <= score <= 39:
            return 'Monitor'
        elif 40 <= score <= 70:
            return 'Stable'
        return 'Unknown'
    df['Risk Category'] = df['Total Score'].apply(get_risk_category)

    # 3. Sentiment Analysis (TextBlob)
    df['Sentiment'] = df['Comments'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    # 4. Anomaly Detection (IsolationForest)
    if len(df) >= 10: # Minimum data points for IsolationForest
        try:
            # Features for anomaly detection
            X_anomaly = df[['Total Score', 'Sentiment']].values
            
            # Train the Isolation Forest model
            iso_forest = IsolationForest(
                contamination='auto', random_state=42
            )
            iso_forest.fit(X_anomaly)
            
            # Predict anomalies (-1 = anomaly, 1 = normal)
            df['Anomaly'] = iso_forest.predict(X_anomaly)
            df['Anomaly'] = df['Anomaly'].map({1: 'No', -1: 'Yes'})
        except Exception as e:
            st.warning(f"Anomaly detection failed: {e}")
            df['Anomaly'] = 'Not Applicable'
    else:
        df['Anomaly'] = 'Requires More Data'
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    return df.sort_values(by='Date', ascending=False)


# --- Tab Functions ---

def tab_upload_data():
    """Tab 1: Upload Data and Initial Processing."""
    st.header("⬆️ Upload Student Data")
    st.markdown("Upload a CSV file to begin analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv"
    )
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['Student', 'Year Group', 'Date', 'Comments', 
                             'Intervention Details'] + CATEGORY_COLS
            if not all(col in df_raw.columns for col in required_cols):
                st.error("The uploaded CSV is missing required columns. Please check the 'Scoring Guide' tab for the full list.")
                st.stop()

            # Process data and store in session state
            st.session_state['df'] = process_data(df_raw)
            st.success("Data successfully uploaded and processed!")
            st.subheader("Processed DataFrame Sample")
            st.dataframe(st.session_state['df'].head())
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    elif not st.session_state['df'].empty:
        st.subheader("Current Loaded Data")
        st.dataframe(st.session_state['df'].head(5))
        st.info(f"Total entries: {len(st.session_state['df'])}")
        
    else:
        st.info("Please upload a CSV file to load data.")


def tab_add_entry():
    """Tab 2: Manually add a new student record."""
    st.header("✍️ Add New Student Entry")
    
    with st.form("new_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            student = st.text_input("Student Name/ID *", key="student_name")
            year_group = st.text_input("Year Group *", key="year_group")
            date_entry = st.date_input("Date of Entry *", key="date_entry")
            
        with col2:
            st.markdown("---")
            st.subheader("Category Scores (1 = Significant Concern, 10 = Strong)")
            scores = {}
            # Use two columns for the sliders for better layout
            sl_col1, sl_col2 = st.columns(2)
            
            for i, col in enumerate(CATEGORY_COLS):
                target_col = sl_col1 if i < (len(CATEGORY_COLS) + 1) // 2 else sl_col2
                scores[col] = target_col.slider(
                    col, 1, 10, 5, key=f"score_{col}"
                )
        
        st.markdown("---")
        comments = st.text_area("Comments", key="comments")
        intervention_details = st.text_area("Intervention Details", key="intervention_details")
        
        submitted = st.form_submit_button("Add Entry and Reprocess Data")

        if submitted:
            if not student or not year_group:
                st.error("Please fill in Student Name/ID and Year Group.")
            else:
                new_entry = {
                    'Student': student,
                    'Year Group': year_group,
                    'Date': pd.to_datetime(date_entry),
                    'Comments': comments,
                    'Intervention Details': intervention_details,
                    **scores
                }
                new_df = pd.DataFrame([new_entry])
                
                # Check if session_state['df'] is empty to concatenate
                if st.session_state['df'].empty:
                    st.session_state['df'] = new_df
                else:
                    st.session_state['df'] = pd.concat(
                        [st.session_state['df'], new_df], 
                        ignore_index=True
                    )
                
                # Reprocess the entire dataset
                st.session_state['df'] = process_data(st.session_state['df'])
                
                st.success(f"Entry for **{student}** added and data reprocessed.")
                st.balloons()

def tab_overview():
    """Tab 3: Key metrics and latest entries."""
    st.header("📊 Student Overview")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return
        
    df = st.session_state['df']
    
    # 1. Metrics
    st.subheader("Key Performance Indicators")
    
    total_students = df['Student'].nunique()
    risk_counts = df['Risk Category'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Unique Students", total_students)
    
    # Safely get counts, defaulting to 0
    high_risk = risk_counts.get('High Risk', 0)
    monitor = risk_counts.get('Monitor', 0)
    stable = risk_counts.get('Stable', 0)
    
    col2.metric("Students: High Risk", high_risk)
    col3.metric("Students: Monitor", monitor)
    col4.metric("Students: Stable", stable)
    
    st.markdown("---")
    
    # 2. Latest Entries
    st.subheader("Latest 10 Entries")
    st.dataframe(
        df[['Date', 'Student', 'Year Group', 'Total Score', 'Risk Category', 'Sentiment', 'Anomaly']]
        .head(10)
        .style.background_gradient(
            subset=['Total Score'], cmap='RdYlGn'
        )
    )

def tab_visuals():
    """Tab 4: Data Visualizations."""
    st.header("📈 Data Visualizations")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return
        
    df = st.session_state['df']

    # 1. Risk Category Distribution (Bar Chart)
    st.subheader("Risk Category Distribution")
    risk_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Risk Category', sort=['High Risk', 'Monitor', 'Stable']),
        y='count()',
        color=alt.Color('Risk Category', scale=alt.Scale(domain=['High Risk', 'Monitor', 'Stable'], range=['red', 'orange', 'green'])),
        tooltip=['Risk Category', 'count()']
    ).properties(
        title="Student Count by Risk Category"
    )
    st.altair_chart(risk_chart, use_container_width=True)
    
    st.markdown("---")

    # 2. Total Score by Student (Bar Chart)
    st.subheader("Total Score by Student (Latest Entry)")
    
    # Get the latest entry for each student
    latest_df = df.sort_values('Date', ascending=False).drop_duplicates(subset=['Student'])

    score_chart = alt.Chart(latest_df).mark_bar().encode(
        x=alt.X('Total Score', title="Total Score (Max 70)"),
        y=alt.Y('Student', sort='-x'),
        color=alt.Color('Risk Category', scale=alt.Scale(domain=['High Risk', 'Monitor', 'Stable'], range=['red', 'orange', 'green'])),
        tooltip=['Student', 'Total Score', 'Risk Category']
    ).properties(
        title="Latest Total Score for Each Student"
    ).interactive()
    st.altair_chart(score_chart, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Sentiment Over Time (Line Chart)
    st.subheader("Sentiment Polarity Over Time")
    
    # Ensure Date is recognized for time series
    df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    sentiment_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Date', axis=alt.Axis(format='%Y-%m-%d')),
        y=alt.Y('Sentiment', title="Comment Sentiment Polarity"),
        color='Student',
        tooltip=['Date_Str', 'Student', 'Sentiment', 'Comments']
    ).properties(
        title="Comment Sentiment Trends"
    ).interactive()
    st.altair_chart(sentiment_chart, use_container_width=True)


def tab_interventions():
    """Tab 5: Filtered table of students with intervention details."""
    st.header("📋 Intervention Tracker")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return

    df = st.session_state['df']
    
    st.sidebar.subheader("Intervention Filters")
    
    # Filters
    year_groups = ['All'] + sorted(df['Year Group'].astype(str).unique().tolist())
    selected_year = st.sidebar.selectbox("Filter by Year Group", year_groups)
    
    risk_categories = ['All'] + ['High Risk', 'Monitor', 'Stable']
    selected_risk = st.sidebar.selectbox("Filter by Risk Category", risk_categories)
    
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['Year Group'] == selected_year]
        
    if selected_risk != 'All':
        filtered_df = filtered_df[filtered_df['Risk Category'] == selected_risk]
        
    
    # Display results
    st.info(f"Showing **{len(filtered_df)}** entries matching the criteria.")
    
    st.dataframe(
        filtered_df[[
            'Date', 'Student', 'Year Group', 'Risk Category', 'Total Score', 'Intervention Details'
        ]].style.background_gradient(
            subset=['Total Score'], cmap='RdYlGn'
        )
    )

def tab_comments():
    """Tab 6: Comments with sentiment analysis and filtering."""
    st.header("💬 Comments and Sentiment")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return

    df = st.session_state['df'].copy()
    
    st.sidebar.subheader("Sentiment Filter")
    
    # Filter by sentiment range (-1.0 to 1.0)
    sentiment_range = st.sidebar.slider(
        "Select Sentiment Range", 
        min_value=-1.0, 
        max_value=1.0, 
        value=(-1.0, 1.0),
        step=0.1
    )
    
    # Apply filter
    filtered_df = df[
        (df['Sentiment'] >= sentiment_range[0]) & 
        (df['Sentiment'] <= sentiment_range[1])
    ]
    
    st.info(f"Showing **{len(filtered_df)}** comments with sentiment between {sentiment_range[0]:.1f} and {sentiment_range[1]:.1f}.")
    
    # Style the sentiment column
    def style_sentiment(val):
        """Color code sentiment."""
        if val < 0:
            return 'color: red'
        elif val > 0:
            return 'color: green'
        return 'color: black'

    st.dataframe(
        filtered_df[['Date', 'Student', 'Year Group', 'Sentiment', 'Comments']]
        .sort_values(by='Sentiment', ascending=True)
        .style.applymap(style_sentiment, subset=['Sentiment'])
    )

def tab_summary():
    """Tab 7: Summary statistics."""
    st.header("✨ Data Summary")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return
        
    df = st.session_state['df']

    # 1. Risk Category Counts
    st.subheader("Risk Category Counts")
    risk_counts = df['Risk Category'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    st.table(risk_counts)

    st.markdown("---")
    
    # 2. Average Scores Across Categories
    st.subheader("Average Scores Across All Entries (Max 10)")
    avg_scores = df[CATEGORY_COLS].mean().sort_values(ascending=False).reset_index()
    avg_scores.columns = ['Category', 'Average Score']
    
    # Style the table
    st.dataframe(
        avg_scores.style.format({'Average Score': "{:.2f}"})
        .background_gradient(subset=['Average Score'], cmap='Blues')
    )

    st.markdown("---")
    
    # 3. Breakdown by Year Group
    st.subheader("Breakdown by Year Group")
    
    # Group by Year Group
    group_summary = df.groupby('Year Group').agg(
        Total_Students=('Student', 'nunique'),
        Avg_Total_Score=('Total Score', 'mean'),
        High_Risk_Count=('Risk Category', lambda x: (x == 'High Risk').sum())
    ).reset_index()
    
    # Style the table
    st.dataframe(
        group_summary.style.format({
            'Avg_Total_Score': "{:.2f}"
        }).background_gradient(
            subset=['High_Risk_Count'], cmap='Reds'
        )
    )

def tab_search():
    """Tab 8: Search functionality."""
    st.header("🔍 Search Records")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return
        
    df = st.session_state['df']
    
    search_term = st.text_input("Enter search term (Student, Comments, Intervention Details)", "").lower()

    if search_term:
        # Search in Student, Comments, and Intervention Details columns
        search_cols = ['Student', 'Comments', 'Intervention Details']
        
        # Create a boolean mask where the search term exists in any of the columns
        mask = df[search_cols].astype(str).apply(
            lambda row: any(search_term in str(x).lower() for x in row), axis=1
        )
        
        results_df = df[mask]
        
        st.info(f"Found **{len(results_df)}** entries matching '{search_term}'.")
        
        st.dataframe(
            results_df[['Date', 'Student', 'Year Group', 'Risk Category', 'Total Score', 'Comments', 'Intervention Details']]
        )
    else:
        st.info("Start typing a term to search across student names, comments, and intervention details.")


def tab_scoring_guide():
    """Tab 9: Documentation and data export."""
    st.header("📚 Scoring Guide and Data Export")

    # 1. Scoring Guidelines
    st.subheader("Individual Category Scoring (1-10)")
    st.markdown(
        """
        | Score Range | Meaning |
        | :--- | :--- |
        | **1 – 3** | **Significant Concern** (Requires immediate action/intense monitoring). |
        | **4 – 6** | **Moderate Concern** (Requires regular monitoring and minor intervention). |
        | **7 – 8** | **Stable** (Doing well, minor checks needed). |
        | **9 – 10**| **Strong** (Excelling, no concerns). |
        """
    )

    st.markdown("---")

    # 2. Category Explanation
    st.subheader("Monitored Categories")
    st.markdown(
        """
        The following 7 categories are scored from 1 to 10. The **Total Score** is the sum of these (Max 70).
        * **Behaviour:** Classroom conduct, following rules, managing emotions.
        * **Home Life:** Stability, support, external factors affecting school.
        * **Eating Habits:** Regularity, healthiness, school lunch participation.
        * **Disabilities:** Impact of any learning/physical disabilities on daily school life.
        * **Interventions:** Success/engagement with current school-based interventions (not the details column).
        * **Safeguarding:** Potential safety risks, vulnerability, or disclosures.
        * **Social:** Peer relationships, social skills, loneliness/isolation.
        """
    )
    
    st.markdown("---")

    # 3. Risk Thresholds
    st.subheader("Risk Thresholds (Total Score)")
    st.markdown(
        """
        | Total Score | Risk Category | Action Implication |
        | :--- | :--- | :--- |
        | **0 – 24** | **High Risk** | **Urgent Action** and case review required. |
        | **25 – 39**| **Monitor** | **Regular Checks** and tailored support plan needed. |
        | **40 – 70**| **Stable** | **Routine Monitoring** and continued positive reinforcement. |
        """
    )
    
    st.markdown("---")

    # 4. Data Export and Sample CSV
    st.subheader("Data Management")

    # Sample CSV for download
    sample_data = {
        'Student': ['Student A', 'Student B'],
        'Year Group': ['Year 7', 'Year 10'],
        'Date': ['2025-10-20', '2025-10-21'],
        'Behaviour': [8, 3], 
        'Home Life': [7, 5], 
        'Eating Habits': [9, 8], 
        'Disabilities': [10, 10], 
        'Interventions': [7, 4], 
        'Safeguarding': [9, 6], 
        'Social': [8, 4], 
        'Comments': ['A great week, very focused in class.', 'Displaying signs of anxiety, needs a check-in.'],
        'Intervention Details': ['None', 'Scheduled for 1:1 mentorship session.']
    }
    sample_df = pd.DataFrame(sample_data)
    
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV Template",
        data=csv,
        file_name='sample_student_tracker_template.csv',
        mime='text/csv',
    )
    
    # Individual student export
    if not st.session_state['df'].empty:
        st.markdown("---")
        st.subheader("Individual Student Data Export")
        df = st.session_state['df']
        student_list = sorted(df['Student'].unique().tolist())
        selected_student = st.selectbox("Select Student to Export", student_list)

        if selected_student:
            student_df = df[df['Student'] == selected_student]
            student_csv = student_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download Data for {selected_student}",
                data=student_csv,
                file_name=f'{selected_student}_behaviour_data.csv',
                mime='text/csv',
            )

def tab_all_insights():
    """Tab 10: Advanced insights like anomalies and classification."""
    st.header("🧠 All Insights and Advanced Analytics")
    
    if st.session_state['df'].empty:
        st.warning("No data loaded. Please upload data or add an entry first.")
        return
        
    df = st.session_state['df']
    
    # 1. Anomalies Flagged
    st.subheader("Anomalies Detected (Isolation Forest)")
    anomaly_df = df[df['Anomaly'] == 'Yes']
    
    if not anomaly_df.empty:
        st.error(f"⚠️ **{len(anomaly_df)}** entries flagged as potential anomalies.")
        st.dataframe(
            anomaly_df[['Date', 'Student', 'Year Group', 'Total Score', 'Sentiment', 'Comments']]
            .sort_values(by='Total Score', ascending=True)
        )
    else:
        st.info("No anomalies detected in the current dataset, or not enough data.")

    st.markdown("---")

    # 2. Top 10 Most Negative Sentiment Comments
    st.subheader("Top 10 Most Negative Sentiment Comments")
    negative_comments = df[['Date', 'Student', 'Sentiment', 'Comments']].sort_values(
        'Sentiment', ascending=True
    ).head(10)
    
    def highlight_negative(val):
        """Color negative sentiment."""
        return 'color: red; font-weight: bold' if val < 0 else ''

    st.dataframe(
        negative_comments.style.applymap(highlight_negative, subset=['Sentiment'])
    )

    st.markdown("---")

    # 3. Risk Category Prediction (Random Forest)
    st.subheader("Risk Category Prediction Model (Random Forest)")
    
    # Check for balanced data (at least 5 samples in each class)
    if len(df['Risk Category'].unique()) == 3 and all(df['Risk Category'].value_counts() >= 5):
        try:
            # Prepare data
            X = df[CATEGORY_COLS].values
            y = df['Risk Category'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            st.success("Random Forest Classifier trained successfully!")
            st.markdown("**Classification Report (Predicting Risk Category)**")
            
            # Display report as a table
            report_df = pd.DataFrame(report).transpose().iloc[:-1, :] # Exclude accuracy/macro avg/weighted avg
            st.dataframe(report_df.style.format("{:.2f}"))
            
            # Feature Importance
            st.markdown("**Feature Importance**")
            feature_imp = pd.Series(model.feature_importances_, index=CATEGORY_COLS).sort_values(ascending=False)
            
            feature_chart = alt.Chart(feature_imp.reset_index()).mark_bar().encode(
                x=alt.X('index', title="Category"),
                y=alt.Y('0', title="Importance Score"),
                tooltip=['index', '0']
            ).properties(title="Importance of Features in Predicting Risk")
            st.altair_chart(feature_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training/evaluating Random Forest: {e}")
            st.info("Ensure all category columns are numeric and data is clean.")
    else:
        st.warning("Not enough balanced data to train the Random Forest Classifier (need at least 5 entries in each of the 3 risk categories).")


def tab_chatbot():
    """Tab 11: Rule-based chatbot."""
    st.header("🤖 Student Data Chatbot")

    if st.session_state['df'].empty:
        st.warning("No data loaded. Chatbot functionality is limited. Please upload data.")
    
    # Dynamic Guidance Panel
    st.sidebar.subheader("Chatbot Guidance")
    if st.session_state['df'].empty:
        st.sidebar.info("Upload data to see example questions.")
    else:
        df = st.session_state['df']
        # Use a real student name for a better example
        student_name_example = df['Student'].iloc[0] if not df['Student'].empty else 'Student A'

        st.sidebar.markdown(
            f"""
            **Example Questions:**
            * Tell me about **High Risk** students.
            * What is the **Average** Total Score?
            * List all **Intervention** details.
            * Which entries are **Anomaly**?
            * Give me a **Summary** of risk categories.
            * What is the score for {student_name_example}?
            """
        )

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about the student data...")
    if prompt:
        # Add user message to history
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response (rule-based)
        response = process_chatbot_query(prompt.lower())

        # Add assistant response to history
        st.session_state['chat_history'].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def process_chatbot_query(query):
    """Rule-based logic for the chatbot."""
    if st.session_state['df'].empty:
        return "I need data loaded to answer specific questions. Please use the 'Upload Data' tab first."

    df = st.session_state['df']
    
    # Define rules and actions
    if "high risk" in query:
        high_risk_students = df[df['Risk Category'] == 'High Risk']['Student'].unique()
        if len(high_risk_students) > 0:
            return "The following students are currently flagged as **High Risk**: " + ", ".join(high_risk_students) + "."
        return "There are currently no students flagged as High Risk."
        
    elif "average" in query or "avg" in query:
        avg_score = df['Total Score'].mean()
        return f"The **average** Total Score across all entries is **{avg_score:.2f}** (out of a maximum of 70)."
        
    elif "intervention" in query or "interventions" in query:
        interventions = df[df['Intervention Details'] != 'None']['Intervention Details'].tolist()
        if len(interventions) > 0:
            return "Here are some of the Intervention Details recorded:\n- " + "\n- ".join(interventions[:5]) + " (and more...)"
        return "No specific intervention details have been recorded (or they are marked as 'None')."

    elif "anomaly" in query:
        anomalies = df[df['Anomaly'] == 'Yes']['Student'].unique()
        if len(anomalies) > 0:
            return f"The following students have entries flagged as **Anomalies**: " + ", ".join(anomalies) + "."
        return "No anomalies have been flagged in the current dataset."

    elif "summary" in query or "risk category" in query:
        risk_counts = df['Risk Category'].value_counts()
        return f"Current **Risk Category Counts**:\n* Stable: {risk_counts.get('Stable', 0)}\n* Monitor: {risk_counts.get('Monitor', 0)}\n* High Risk: {risk_counts.get('High Risk', 0)}"

    # Check for specific student name in the query
    student_list = df['Student'].unique().tolist()
    for student in student_list:
        if student.lower() in query:
            student_data = df[df['Student'] == student].sort_values('Date', ascending=False).iloc[0]
            score = student_data['Total Score']
            risk = student_data['Risk Category']
            return f"The latest entry for **{student}** shows a **Total Score** of **{score}** (Max 70), placing them in the **{risk}** risk category."
            
    return "I'm a rule-based chatbot and can only answer questions related to: **High Risk**, **Average** score, **Intervention** details, **Anomaly** entries, or a general **Summary**. You can also ask for an individual student's score."


# --- Main App Execution ---

st.title("Student Behaviour Tracker 🎓")

# Create the tabs
tab_names = [
    "Upload Data", "Add Entry", "Overview", "Visuals", 
    "Interventions", "Comments", "Summary", "Search", 
    "Scoring Guide", "All Insights", "Chatbot"
]

tabs = st.tabs(tab_names)

# Map tabs to functions
with tabs[0]: tab_upload_data()
with tabs[1]: tab_add_entry()
with tabs[2]: tab_overview()
with tabs[3]: tab_visuals()
with tabs[4]: tab_interventions()
with tabs[5]: tab_comments()
with tabs[6]: tab_summary()
with tabs[7]: tab_search()
with tabs[8]: tab_scoring_guide()
with tabs[9]: tab_all_insights()
with tabs[10]: tab_chatbot()