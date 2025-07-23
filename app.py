import streamlit as st
import pandas as pd
import cloudpickle
import altair as alt
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

alt.data_transformers.enable('default', max_rows=None)

st.markdown("""
    <style>
        /* --- Global Font & Background --- */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f7fa;
        }

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {
            background-color: #ecf0f1;
            border-right: 1px solid #dcdcdc;
        }

        /* --- Titles --- */
        h1, h2, h3, h4 {
            color: #2c3e50;
        }

        /* --- Buttons --- */
        .stButton > button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }

        /* --- Metrics --- */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        /* --- Form labels --- */
        label {
            font-weight: bold;
            color: #34495e;
        }

        /* --- Animations --- */
        .fade-in {
            animation: fadeIn ease 1s;
            -webkit-animation: fadeIn ease 1s;
        }

        @keyframes fadeIn {
            0% {opacity:0;}
            100% {opacity:1;}
        }

        /* --- Table Styling --- */
        .element-container iframe {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA AND MODEL LOADING ---
@st.cache_data
def load_data(path):
    """Loads the car price dataset from a CSV file."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please make sure it's in the same directory as the app.")
        return None

@st.cache_resource
def load_model(path="car_price_model.pkl"):
    """Loads the ML pipeline saved with cloudpickle."""
    try:
        with open(path, "rb") as f:
            model = cloudpickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the data and model
df = load_data('car-price_cleaned.csv')

model = load_model()

if df is None or model is None:
    st.stop()


# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Seller Dashboard", "Buyer Dashboard", "Market Analysis & Insights"])

# --- SHARED DATA for UI ---
brands = sorted(df['Brand'].unique())
models = sorted(df['Model'].unique())
fuel_types = sorted(df['Fuel_Type'].unique())
transmissions = sorted(df['Transmission'].unique())
min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
min_price, max_price = int(df['Price'].min()), int(df['Price'].max())


# =====================================================================================
# --- SELLER DASHBOARD (VERSION AMÉLIORÉE AVEC STRATÉGIE DE PRIX) ---
# =====================================================================================
if page == "Seller Dashboard":
    st.markdown("""
        <style>
            .main-title {
                font-size: 38px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 0.5em;
            }
            .subtitle {
                font-size: 16px;
                color: #7f8c8d;
                text-align: center;
                margin-bottom: 2em;
            }
            .form-box {
                background-color: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #ddd;
                margin-bottom: 25px;
            }
            .price-card {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 12px;
                border-left: 6px solid #27ae60;
                margin-bottom: 20px;
            }
            .price-card h4 {
                color: #2c3e50;
                margin-bottom: 8px;
            }
            .price-value {
                font-size: 32px;
                font-weight: bold;
                color: #27ae60;
            }
            .strategy-box {
                background-color: #e9f5ec;
                border: 1px solid #a3d9b1;
                padding: 15px;
                border-radius: 10px;
                margin-top: 10px;
            }
            .strategy-box h4 {
                color: #155724;
                margin-bottom: 10px;
            }
            .price-tier {
                padding: 10px;
                margin: 8px 0;
                border-left: 5px solid;
                border-radius: 6px;
            }
            .price-tier.max { border-color: #28a745; background-color: #f0fff3; }
            .price-tier.target { border-color: #007bff; background-color: #f0f7ff; }
            .price-tier.quick { border-color: #fd7e14; background-color: #fff8f0; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">📈 Seller Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Get a smart price recommendation and learn how to sell strategically.</div>', unsafe_allow_html=True)

    with st.container():
        with st.expander("📋 Fill in your car details", expanded=True):
            st.markdown('<div class="form-box">', unsafe_allow_html=True)
            with st.form(key='seller_form'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    brand = st.selectbox("Brand", brands, key='brand')
                    year = st.slider("Year", min_year, max_year, max_year - 5, key='year')
                    mileage = st.slider("Mileage (km)", 0, 400000, 100000, 1000, key='mileage')
                with col2:
                    model_car = st.selectbox("Model", models, key='model')
                    engine_size = st.slider("Engine Size (L)", 1.0, 6.0, 2.0, 0.1, key='engine')
                    doors = st.select_slider("Doors", options=[2, 3, 4, 5], value=4, key='doors')
                with col3:
                    fuel_type = st.selectbox("Fuel Type", fuel_types, key='fuel')
                    transmission = st.selectbox("Transmission", transmissions, key='transmission')
                    owner_count = st.select_slider("Previous Owners", options=[1, 2, 3, 4, 5], key='owners')

                submit_button = st.form_submit_button(label='Estimate My Car\'s Price')
            st.markdown('</div>', unsafe_allow_html=True)

    if submit_button:
input_data = {
            'Brand': brand, 'Model': model_car, 'Year': year, 'Engine_Size': engine_size,
            'Fuel_Type': fuel_type, 'Transmission': transmission, 'Mileage': mileage,
            'Doors': doors, 'Owner_Count': owner_count
        }
expected_cols = ['Brand', 'Model', 'Year', 'Engine_Size', 'Fuel_Type', 
                 'Transmission', 'Mileage', 'Doors', 'Owner_Count']

input_df = pd.DataFrame([input_data])[expected_cols]

        predicted_price = model.predict(input_df)[0]

        max_value = predicted_price * 1.05
        quick_sale = predicted_price * 0.80
        acceptance_threshold = predicted_price * 0.95

        img_col, result_col = st.columns([1, 2])
        with img_col:
            try:
                logo_path = f"logos/{brand.lower()}.png"
                st.image(logo_path, use_column_width=True, caption=f"{brand} {model_car}")
            except Exception:
                st.write("")

        with result_col:
            st.markdown(f"""
                <div class="price-card">
                    <h4>Recommended Listing Price</h4>
                    <div class="price-value">€ {predicted_price:,.0f}</div>
                    <p style="font-size:14px; color:#666;">Expected time to sell: 20–35 days</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="strategy-box">
                    <h4>💡 Pricing Strategy</h4>
                    <div class="price-tier max">
                        <strong>Max Value:</strong> € {max_value:,.0f} <br><span>Start here for negotiation</span>
                    </div>
                    <div class="price-tier target">
                        <strong>Target Price:</strong> € {predicted_price:,.0f} <br><span>Likely to sell at this price</span>
                    </div>
                    <div class="price-tier quick">
                        <strong>Quick Sale:</strong> € {quick_sale:,.0f} <br><span>If you need to sell fast</span>
                    </div>
                    <hr style="margin-top:10px; margin-bottom:10px;">
                    <ul>
                        <li>Accept offers above <strong>€ {acceptance_threshold:,.0f}</strong>.</li>
                        <li>Include service history and clean photos.</li>
                        <li>List on multiple platforms for visibility.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🧠 Auto-Generated Sales Pitch"):
            pitch = f"""
            This {year} {brand} {model_car} offers a solid {engine_size:.1f}L engine and a {transmission.lower()} transmission – ideal for smooth daily drives.
            It runs on {fuel_type.lower()} and has covered {mileage:,} km, making it a reliable choice.
            With only {owner_count} previous owner(s) and {doors} doors, this vehicle combines efficiency, comfort, and resale potential.
            """
            st.write(pitch)
    
        st.markdown("---")
        st.subheader("📊 What Drives Your Car's Price?")
        try:
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            coefficients = model.named_steps['regressor'].coef_

            # Création du DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefficients)
            })

            # Nettoyer les noms (ex: num__Engine_Size → Engine Size)
            importance_df['Feature'] = importance_df['Feature'].str.replace(r'^.*?__', '', regex=True).str.replace('_', ' ')

            # Pourcentages
            importance_df['Percent'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()
            importance_df = importance_df.sort_values(by='Percent', ascending=False).head(10)
            importance_df['Label'] = importance_df['Percent'].apply(lambda x: f"{x:.1f}%")

            # Graphe Plotly
            fig_importance = px.bar(
                importance_df,
                x='Percent',
                y='Feature',
                orientation='h',
                text='Label',
                title='Top 10 Features Influencing Price (in %)',
                color='Percent',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

            # Résumé automatique
            top_features = importance_df.head(3)
            feature_summary = ", ".join(top_features['Feature'].values)
            total_weight = top_features['Percent'].sum()

            st.info(f"🔎 **Top drivers:** {feature_summary} account for **{total_weight:.1f}%** of the price prediction model’s influence.")

            st.caption("Feature importance is based on the normalized absolute weights from the linear regression model.")
        except Exception as e:
            st.warning("⚠️ Could not generate the feature importance chart.")


# =====================================================================================
# --- BUYER DASHBOARD (Mise à jour avec la jauge) ---
# =====================================================================================
elif page == "Buyer Dashboard":
    st.markdown("""
        <style>
            .main-title {
                font-size: 38px;
                font-weight: bold;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 0.3em;
            }
            .subtitle {
                font-size: 16px;
                color: #7f8c8d;
                text-align: center;
                margin-bottom: 2em;
            }
            .form-box {
                background-color: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #ddd;
                margin-bottom: 25px;
            }
            .predict-card {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 6px solid #3498db;
                margin-bottom: 20px;
            }
            .predict-card h4 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .predict-card .price {
                font-size: 32px;
                font-weight: bold;
                color: #27ae60;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🛒 Buyer Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estimate a fair price for a car and explore inventory options</div>', unsafe_allow_html=True)

    with st.container():
        with st.expander("🔍 Estimate Price of a Specific Car", expanded=True):
            st.markdown('<div class="form-box">', unsafe_allow_html=True)
            with st.form(key='buyer_estimator_form'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    brand_b = st.selectbox("Brand", brands, key='brand_b')
                    year_b = st.slider("Year", min_year, max_year, max_year - 5, key='year_b')
                    mileage_b = st.slider("Mileage (km)", 0, 400000, 100000, 1000, key='mileage_b')
                with col2:
                    model_car_b = st.selectbox("Model", models, key='model_b')
                    engine_size_b = st.slider("Engine Size (L)", 1.0, 6.0, 2.0, 0.1, key='engine_b')
                    doors_b = st.select_slider("Doors", options=[2, 3, 4, 5], value=4, key='doors_b')
                with col3:
                    fuel_type_b = st.selectbox("Fuel Type", fuel_types, key='fuel_b')
                    transmission_b = st.selectbox("Transmission", transmissions, key='transmission_b')
                    owner_count_b = st.select_slider("Previous Owners", options=[1, 2, 3, 4, 5], key='owners_b')

                submit_button_b = st.form_submit_button(label='Estimate Fair Price')
            st.markdown('</div>', unsafe_allow_html=True)

    if submit_button_b:
        input_data_b = {
            'Brand': brand_b, 'Model': model_car_b, 'Year': year_b, 'Engine_Size': engine_size_b,
            'Fuel_Type': fuel_type_b, 'Transmission': transmission_b, 'Mileage': mileage_b,
            'Doors': doors_b, 'Owner_Count': owner_count_b
        }
        input_df_b = pd.DataFrame([input_data_b])
        predicted_price_b = model.predict(input_df_b)[0]

        img_col, data_col = st.columns([1, 2])
        with img_col:
            try:
                logo_path = f"logos/{brand_b.lower()}.png"
                st.image(logo_path, use_column_width=True)
            except FileNotFoundError:
                st.write("")

        with data_col:
            st.markdown(f"""
                <div class="predict-card">
                    <h4>Estimated Fair Price</h4>
                    <div class="price">€ {predicted_price_b:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)

            similar_cars = df[
                (df['Brand'] == brand_b) &
                (df['Model'] == model_car_b) &
                (df['Year'].between(year_b - 2, year_b + 2))
            ]
            num_similar = len(similar_cars)

            if num_similar >= 10:
                reliability_score = "✅ High"
                recommendation = "Price supported by many similar cars."
            elif 1 < num_similar < 10:
                reliability_score = "⚠️ Medium"
                recommendation = "Estimate based on limited matches. Proceed with caution."
            else:
                reliability_score = "❌ Low"
                recommendation = "Very limited data. Price might vary significantly."

            st.info(f"**Reliability:** {reliability_score}")
            st.caption(f"💡 {recommendation}")

        st.markdown("---")
        st.subheader("📈 Visual Analysis")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if not similar_cars.empty:
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=similar_cars['Price'], name='Similar Cars', marker_color='royalblue'))
                fig_box.add_hline(y=predicted_price_b, line_dash="dot",
                                  annotation_text="Predicted Price", line_color="red")
                fig_box.update_layout(title="Price Range of Similar Cars")
                st.plotly_chart(fig_box, use_container_width=True)

        with chart_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price_b,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [min_price, max_price]},
                    'bar': {'color': "#2980b9"},
                    'steps': [
                        {'range': [min_price, df['Price'].quantile(0.33)], 'color': '#ecf0f1'},
                        {'range': [df['Price'].quantile(0.33), df['Price'].quantile(0.66)], 'color': '#bdc3c7'},
                        {'range': [df['Price'].quantile(0.66), max_price], 'color': '#95a5a6'}
                    ]
                }
            ))
            fig_gauge.update_layout(title="Market Position", height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

        if not similar_cars.empty:
            with st.expander("📋 See Similar Cars"):
                st.dataframe(similar_cars[['Brand', 'Model', 'Year', 'Mileage', 'Price']].sort_values('Price'), use_container_width=True)

    # === INVENTORY EXPLORER ===
    st.markdown("---")
    st.subheader("🔎 Explore Available Inventory")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        brand_filter = st.multiselect("Brand(s)", brands, default=brands[:3])
    with f_col2:
        price_filter = st.slider("Max Price (€)", min_price, max_price, max_price, 500)
    with f_col3:
        year_filter = st.slider("Min Year", min_year, max_year, max_year - 10)
    with f_col4:
        mileage_filter = st.slider("Max Mileage (km)", 0, 400000, 200000, 1000)

    if not brand_filter:
        brand_filter = brands

    filtered_df = df[
        (df['Brand'].isin(brand_filter)) &
        (df['Price'] <= price_filter) &
        (df['Year'] >= year_filter) &
        (df['Mileage'] <= mileage_filter)
    ]
    st.markdown(f"**✅ {len(filtered_df)} car(s) found matching your filters**")

    if filtered_df.empty:
        st.warning("No matching cars. Try relaxing your filters.")
    else:
        st.dataframe(filtered_df[['Brand', 'Model', 'Year', 'Price', 'Mileage']], use_container_width=True, height=350)


# =====================================================================================
# --- MARKET ANALYSIS & INSIGHTS (VERSION CORRIGÉE ET AMÉLIORÉE) ---
# =====================================================================================
elif page == "Market Analysis & Insights":
    # ===== CSS pour le design pro =====
    st.markdown("""
        <style>
            .main-title {
                font-size: 38px;
                font-weight: 700;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 0.5em;
            }
            .subtitle {
                font-size: 16px;
                color: #7f8c8d;
                text-align: center;
                margin-bottom: 2em;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                text-align: center;
            }
            .metric-card h4 {
                margin: 0;
                color: #2980b9;
            }
            .metric-card p {
                margin: 5px 0 0;
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
            }
            .filter-box {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #2980b9;
            }
        </style>
    """, unsafe_allow_html=True)

    # ===== TITRES =====
    st.markdown('<div class="main-title">🌍 Market Analysis & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore the second-hand car market interactively based on filters</div>', unsafe_allow_html=True)

    # ===== FILTRE =====
    with st.container():
        with st.expander("🔍 Click to Filter the Market", expanded=True):
            st.markdown('<div class="filter-box">', unsafe_allow_html=True)
            selected_brands = st.multiselect("Choose Brand(s)", options=brands, default=brands[:3])
            st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df[df['Brand'].isin(selected_brands)] if selected_brands else df.copy()

    # ===== METRICS CARDS =====
    avg_price = int(filtered_df['Price'].mean())
    avg_mileage = int(filtered_df['Mileage'].mean())
    car_count = len(filtered_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h4>Total Listings</h4><p>{:,}</p></div>'.format(car_count), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Average Price</h4><p>€ {:,}</p></div>'.format(avg_price), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Average Mileage</h4><p>{:,} km</p></div>'.format(avg_mileage), unsafe_allow_html=True)

    st.markdown("---")

    # ===== GRAPHIQUES =====
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.subheader("📊 Average Price by Brand")
        avg_price_brand = filtered_df.groupby('Brand')['Price'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_price_brand, x='Brand', y='Price', color='Brand',
                     title="Average Price per Brand",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⛽ Market Share by Fuel Type")
        fuel_counts = filtered_df['Fuel_Type'].value_counts().reset_index()
        fuel_counts.columns = ['Fuel_Type', 'count']
        fig = px.pie(fuel_counts, names='Fuel_Type', values='count',
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with vis_col2:
        st.subheader("📈 Price Trend by Year")
        avg_price_year = filtered_df.groupby('Year')['Price'].mean().reset_index()
        fig = px.line(avg_price_year, x='Year', y='Price', markers=True,
                      title="Price Evolution Over Years",
                      color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🚘 Mileage vs. Price")
        sample_size = min(1000, len(filtered_df))
        fig = px.scatter(filtered_df.sample(n=sample_size, random_state=42),
                         x='Mileage', y='Price',
                         opacity=0.5,
                         color='Brand',
                         title="Correlation: Mileage and Price",
                         hover_data=['Brand', 'Model', 'Year'])
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
