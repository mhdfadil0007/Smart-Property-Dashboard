import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

# -------------------
# Page setup
# -------------------
st.set_page_config(page_title="Smart Property Insights", layout="wide")

# -------------------
# Load & Clean Data
# -------------------
@st.cache_data
def load_properties():
    file = "dubai_properties.csv"
    df = pd.read_csv(file)

    # Standardize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Ensure Property Id exists
    if "Property Id" not in df.columns:
        df.insert(0, "Property Id", [f"AUTO{i+1}" for i in range(len(df))])

    # Add 10 new rows safely
    new_data = [
        {"Property Id":"AUTO1","Location":"Downtown Dubai","Price":1500000,"Area_In_Sqft":1200,"Beds":2,"Baths":2,"Owner":"Smart Properties LLC"},
        {"Property Id":"AUTO2","Location":"Dubai Marina","Price":2300000,"Area_In_Sqft":1500,"Beds":3,"Baths":3,"Owner":"Marina Heights Real Estate"},
        {"Property Id":"AUTO3","Location":"Business Bay","Price":1100000,"Area_In_Sqft":950,"Beds":1,"Baths":1,"Owner":"Future Homes Realty"},
        {"Property Id":"AUTO4","Location":"Palm Jumeirah","Price":6800000,"Area_In_Sqft":3500,"Beds":4,"Baths":5,"Owner":"Ocean View Properties"},
        {"Property Id":"AUTO5","Location":"Jumeirah Village Circle","Price":950000,"Area_In_Sqft":800,"Beds":1,"Baths":1,"Owner":"Urban Nest Realty"},
        {"Property Id":"AUTO6","Location":"Arabian Ranches","Price":3200000,"Area_In_Sqft":2800,"Beds":4,"Baths":4,"Owner":"Desert Sands Real Estate"},
        {"Property Id":"AUTO7","Location":"Al Barsha","Price":1800000,"Area_In_Sqft":1600,"Beds":3,"Baths":3,"Owner":"Cityscape Properties"},
        {"Property Id":"AUTO8","Location":"Meydan","Price":4500000,"Area_In_Sqft":3100,"Beds":5,"Baths":5,"Owner":"Prime Location Realty"},
        {"Property Id":"AUTO9","Location":"Downtown Dubai","Price":2600000,"Area_In_Sqft":1450,"Beds":2,"Baths":2,"Owner":"Smart Properties LLC"},
        {"Property Id":"AUTO10","Location":"Dubai Hills","Price":5900000,"Area_In_Sqft":4000,"Beds":5,"Baths":6,"Owner":"Green Valley Properties"},
    ]
    new_df = pd.DataFrame(new_data)

    # Align columns
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    new_df = new_df[df.columns]

    df = pd.concat([df, new_df], ignore_index=True)

    # Convert numeric columns
    numeric_cols = ["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical info
    required_cols = ["Location", "Type"]
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])

    return df


@st.cache_data
def load_transactions():
    df = pd.read_csv("dubai_transactions.csv")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["year"] = df["Transaction Date"].dt.year
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Area"] = df["Area"].astype(str).str.strip()
    df = df.dropna(subset=["year", "Amount", "Area"])
    return df


with st.spinner("Loading data..."):
    properties = load_properties()
    transactions = load_transactions()

# -------------------
# Compute Area Growth
# -------------------
@st.cache_data
def compute_area_growth(transactions):
    growth_list = []
    for loc, grp in transactions.groupby("Area"):
        grp_sorted = grp.sort_values("year")
        if len(grp_sorted) >= 2:
            first = grp_sorted["Amount"].iloc[0]
            last = grp_sorted["Amount"].iloc[-1]
            growth_pct = (last - first) / first * 100 if first > 0 else 0
            avg_price = grp_sorted["Amount"].mean()
            growth_list.append({
                "Area": loc,
                "Growth %": round(growth_pct, 2),
                "Avg Price": round(avg_price, 2)
            })
    growth_df = pd.DataFrame(growth_list).sort_values("Growth %", ascending=False)
    return growth_df

growth_df = compute_area_growth(transactions)

# -------------------
# Sidebar Inputs
# -------------------
with st.sidebar:
    st.header("🏠 Property Inputs")

    all_locations = ["Select Location"] + sorted(properties["Location"].unique())
    location = st.selectbox("Select Location", all_locations, index=0)

    all_types = ["Select Property Type"] + list(properties["Type"].unique())
    ptype = st.selectbox("Select Property Type", all_types, index=0)

    size = st.number_input("Size (sqft or rooms)", min_value=100, max_value=10000, step=100, value=1000)
    price = st.number_input("Estimated Property Value (AED)", min_value=0, step=50000, value=1000000)
    rent = st.number_input("Annual Rent (AED)", min_value=0, step=10000, value=100000)
    budget = st.number_input("Budget (AED)", min_value=50000, step=50000, value=1200000)

    all_req_types = ["Select Property Type"] + list(properties["Type"].unique())
    req_type = st.selectbox("Preferred Property Type", all_req_types, index=0)

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Market Trends", "Recommendations", "Transactions", "Tools"
])

# -------------------
# Tab 1 — Overview
# -------------------
with tab1:
    if location == "Select Location" or ptype == "Select Property Type":
        st.image("https://firststepproperties.ae/wp-content/uploads/2023/03/home-banner.jpg", use_column_width=True)
        st.markdown("<h1 style='text-align:center;color:#003366;'>Welcome to First Step Properties LLC</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;color:#0055a5;'>Start by selecting a Location and Property Type from the sidebar.</h3>", unsafe_allow_html=True)
    else:
        st.header("📊 Core Insights")

        avg_rent_row = properties[properties["Location"].str.strip().str.lower() == location.strip().lower()]

        if not avg_rent_row.empty:
            if price > 0 and rent > 0:
                rental_yield = (rent / price) * 100
                st.metric("Rental Yield", f"{rental_yield:.2f}%")

                avg_rent = avg_rent_row["Rent"].mean()
                pct_diff = (rent - avg_rent) / avg_rent * 100
                st.write(f"💰 *Average Market Rent in {location}:* {avg_rent:,.0f} AED")
                st.write(f"🔍 Rent difference vs market: {pct_diff:.2f}%")

                if rent > avg_rent * 1.15:
                    st.error("⚠ This property seems overpriced.")
                else:
                    st.success("✅ This property is fairly priced.")
            else:
                st.info("Enter Property Value and Annual Rent to compute rental yield.")

            # Rent distribution histogram
            fig_dist = px.histogram(avg_rent_row, x="Rent", nbins=30, title=f"Rent Distribution in {location}",
                                    color_discrete_sequence=["lightblue"], hover_data=["Beds", "Baths", "Type"])
            if rent > 0:
                fig_dist.add_vline(x=rent, line_dash="dash", line_color="red",
                                    annotation_text="Your Rent", annotation_position="top left")
            st.plotly_chart(fig_dist, use_container_width=True)

 # --- HOTSPOT FINDER in Overview ---
st.subheader("🏘️ Hotspot Finder")
numeric_cols = ["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]
numeric_cols = [col for col in numeric_cols if col in properties.columns]
cluster_df = properties[numeric_cols].dropna()

if len(cluster_df) >= 5:
    k = min(5, len(cluster_df))
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_df["Cluster"] = kmeans.fit_predict(cluster_df[numeric_cols])

    # Hover columns only if they exist
    hover_cols = [col for col in ["Beds", "Baths", "Price"] if col in cluster_df.columns]

    fig_hotspot = px.scatter(cluster_df, x="Area_In_Sqft", y="Rent",
                             color="Cluster", hover_data=hover_cols,
                             title="AI Hotspot Finder: Rent vs Area")
    st.plotly_chart(fig_hotspot, use_container_width=True)
else:
    st.info("Not enough data to perform hotspot clustering.")


# -------------------
# Tab 2 — Market Trends
# -------------------
with tab2:
    st.header("🤖 AI Market Intelligence (AMI)")
    st.subheader("📊 AI Narrative Insights")
    st.write("""
    This tab shows AI-powered market trends:
    - **Hotspots & Risk Zones:** Areas with highest and lowest price growth.
    - **Forecast:** Projected property value for next year based on past trends.
    - **Early Warning Signals:** Detect sudden market surges or drops.
    """)

    if not growth_df.empty:
        top_growth = growth_df.iloc[0]
        bottom_growth = growth_df.iloc[-1]

        st.success(f"🔥 {top_growth['Area']} is the fastest growing area (+{top_growth['Growth %']}%). Future investment hotspot.")
        st.warning(f"⚠️ {bottom_growth['Area']} shows the weakest trend ({bottom_growth['Growth %']}%). Risk zone.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("🏆 Hotspots (Top 3)")
            st.table(growth_df.head(3))
        with col2:
            st.write("🚨 Risk Zones (Bottom 3)")
            st.table(growth_df.tail(3))

        selected_area = st.selectbox("Choose an Area for Forecast", growth_df["Area"].unique())
        loc_data = transactions[transactions["Area"].str.lower() == selected_area.lower()].sort_values("year")

        if len(loc_data) >= 2:
            fig_forecast = px.line(loc_data, x="year", y="Amount", markers=True,
                                   title=f"Transactions in {selected_area} with Forecast",
                                   hover_data=["Amount"])
            last_year = loc_data["year"].iloc[-1]
            last_amount = loc_data["Amount"].iloc[-1]
            prev_amount = loc_data["Amount"].iloc[-2]
            growth_rate = (last_amount - prev_amount) / prev_amount if prev_amount > 0 else 0
            next_year = last_year + 1
            forecast_amount = last_amount * (1 + growth_rate)

            fig_forecast.add_scatter(x=[next_year], y=[forecast_amount], mode="markers+text",
                                     text=["Forecast"], name="Forecast",
                                     marker=dict(color="red", size=12))
            st.plotly_chart(fig_forecast, use_container_width=True)

            if growth_rate > 0.05:
                st.success(f"{selected_area} likely remains an investment hotspot next year.")
            elif growth_rate > 0:
                st.info(f"{selected_area} shows steady growth. Good for rental investors.")
            else:
                st.error(f"{selected_area} may stagnate or decline. Consider carefully.")
        else:
            st.info("Not enough data for forecasting this area.")

        st.subheader("⚡ Early Warning Signals")
        sudden_shifts = growth_df[growth_df["Growth %"].abs() > 20]
        if not sudden_shifts.empty:
            for _, row in sudden_shifts.iterrows():
                if row["Growth %"] > 0:
                    st.success(f"🚀 {row['Area']} is experiencing a sudden surge (+{row['Growth %']}%). Emerging opportunity.")
                else:
                    st.error(f"📉 {row['Area']} is experiencing a sharp drop ({row['Growth %']}%). Warning sign.")
        else:
            st.write("✅ No major market shocks detected in recent years.")

# -------------------
# Tab 3 — Recommendations
# -------------------
with tab3:
    st.header("🤝 AI-Powered Property Recommendations")
    st.write("""
    AI-powered recommendations based on:
    - **Rental Yield:** Potential return on investment.
    - **Area Growth %:** Historical growth trends.
    - **AI Score:** Weighted score combining yield and growth.
    - Recommendations: **Invest / Hold / Sell** based on AI Score.
    """)

    @st.cache_data
    def compute_recommendations(df_props, df_growth, budget, req_type):
        df = df_props.copy()
        if req_type == "Select Property Type":
            matches = df[df["Rent"] <= budget].copy()
        else:
            matches = df[(df["Rent"] <= budget) & (df["Type"].str.strip().str.lower() == req_type.strip().lower())].copy()

        if matches.empty:
            return pd.DataFrame()

        matches["Rental Yield %"] = (matches["Rent"] / matches["Area_In_Sqft"]) * 100
        area_growth = {row["Area"]: row["Growth %"] for _, row in df_growth.iterrows()}
        matches["Area Growth %"] = matches["Location"].map(area_growth).fillna(0)

        matches["AI Score"] = matches["Rental Yield %"] * 0.5 + matches["Area Growth %"] * 0.5

        def recommend(score):
            if score >= 60:
                return "Invest"
            elif score >= 40:
                return "Hold"
            else:
                return "Sell"

        matches["Recommendation"] = matches["AI Score"].apply(recommend)
        matches["Next Year Forecast"] = matches["Rent"] * 1.05
        matches["Insight"] = matches.apply(lambda r: f"AI Score {r['AI Score']:.1f}. Recommended to {r['Recommendation']}", axis=1)
        return matches

    matches = compute_recommendations(properties, growth_df, budget, req_type)

    if not matches.empty:
        st.subheader("📌 Recommendations Table")
        st.dataframe(matches[["Location","Type","Beds","Baths","Rent","Rental Yield %",
                              "Area Growth %","AI Score","Next Year Forecast",
                              "Recommendation","Insight"]].head(50))

        st.subheader("🗺️ Property Map by AI Score")
        if "Latitude" in matches.columns and "Longitude" in matches.columns:
            st.map(matches.rename(columns={"Latitude":"lat","Longitude":"lon"}).dropna(subset=["lat","lon"]))

        st.subheader("📊 Investment Radar (AI Score vs Rental Yield)")
        fig_radar = px.scatter(matches, x="Rental Yield %", y="AI Score",
                               color="Recommendation", size="Beds",
                               hover_data=["Location","Type","Rent","AI Score","Area Growth %"],
                               title="Investment Radar: Rental Yield vs AI Score")
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.write("No matches found — expand budget or change type.")

# -------------------
# Tab 4 — Transactions
# -------------------
with tab4:
    st.header("📊 Transaction Trends Over Time")
    st.write("""
    Transaction trends over years:
    - Shows price changes for properties in the selected location.
    - Helps identify growth or decline in market value.
    """)

    available_locations = sorted(transactions["Area"].str.strip().unique())
    selected_location = st.selectbox("Select location with transaction history", available_locations)

    loc_data = transactions[transactions["Area"].str.strip().str.lower() == selected_location.strip().lower()].sort_values("year")

    if not loc_data.empty:
        fig_trans = px.line(loc_data, x="year", y="Amount", markers=True,
                            title=f"Transaction Trends in {selected_location}",
                            hover_data=["Amount"])
        st.plotly_chart(fig_trans, use_container_width=True)
    else:
        st.info("No transaction history for this location.")

## -------------------
# Tab 5 — Tools
# -------------------
with tab5:
    st.header("🛠️ Tools")
    st.write("""
    Tools available:
    - Chatbot: Ask questions about average rents or investment insights.
    - SPA Upload: Extract structured details from your SPA document.
    - Search: Find properties in the dataset by ID or reference, with AI suggestions.
    """)

    # ---- Chatbot ----
    query = st.text_input("Ask the bot a question (e.g., 'avg rent in Yas Island?')")
    if query:
        answered = False
        for loc in properties["Location"].unique():
            if loc.lower() in query.lower():
                avg_rent = properties[properties["Location"] == loc]["Rent"].mean()
                area_growth = growth_df[growth_df["Area"].str.lower() == loc.lower()]["Growth %"].values
                growth_info = f" Growth: {area_growth[0]:.2f}%." if len(area_growth) > 0 else ""
                st.write(f"Bot: Average rent in {loc} is {avg_rent:,.0f} AED.{growth_info}")
                answered = True
                break
        if not answered:
            st.write("Bot: This feature is in preview. Try a location name in your query.")

    # ---- SPA Upload ----
    uploaded = st.file_uploader("Upload SPA (txt)", type=["txt"])
    if uploaded:
        txt = uploaded.read().decode("utf-8")
        st.write("### SPA Raw Text")
        st.text_area("Document content", txt, height=200)
        extracted_data = []
        for line in txt.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                extracted_data.append({"Field": key.strip(), "Value": val.strip()})
        if extracted_data:
            st.write("### Structured SPA Property Details")
            num_properties = len(extracted_data) // 5
            for i in range(num_properties):
                st.subheader(f"🏠 Property {i+1}")
                st.table(pd.DataFrame(extracted_data[i*5:(i+1)*5]))

    # ---- Property Search ----
    if "Property Id" in properties.columns:
        search_id = st.text_input("🔎 Search Property ID", "")
        if search_id:
            search_result = properties[properties["Property Id"].astype(str).str.contains(search_id, case=False, na=False)]
            if not search_result.empty:
                st.write("### Search Result")
                st.dataframe(search_result)
            else:
                st.write("No property found with this ID.")
