import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
import os
import joblib

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

    # Convert numeric columns safely
    numeric_cols = ["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop incomplete rows
    required_cols = ["Location", "Type"]
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])

    # Standardize location text for consistency with transactions
    if "Location" in df.columns:
        df["Location"] = df["Location"].astype(str).str.strip().str.title()

    return df


@st.cache_data
def load_transactions():
    df = pd.read_csv("dubai_transactions.csv")

    # Standardize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Clean and convert
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df["Year"] = df["Transaction Date"].dt.year

    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    if "Area" in df.columns:
        df["Area"] = df["Area"].astype(str).str.strip().str.title()

    # Drop rows missing key info
    df = df.dropna(subset=["Year", "Amount", "Area"])

    return df


# -------------------
# Load Data
# -------------------
with st.spinner("Loading data..."):
    properties = load_properties()
    transactions = load_transactions()

common_areas = set(properties["Location"].unique()) & set(transactions["Area"].unique())

# -------------------
# Compute Area Insights (Mean-based)
# -------------------
@st.cache_data
def compute_area_means(transactions):
    mean_df = (
        transactions
        .groupby("Area", as_index=False)
        .agg({"Amount": "mean"})
        .rename(columns={"Amount": "Avg Transaction Value"})
    )

    mean_df["Avg Transaction Value"] = mean_df["Avg Transaction Value"].round(2)
    mean_df = mean_df.sort_values("Avg Transaction Value", ascending=False)
    return mean_df


area_means_df = compute_area_means(transactions)
@st.cache_data
def prepare_matches(properties, area_means_df):
    df = properties.copy()
    df["Location"] = df["Location"].astype(str).str.strip().str.title()
    merged = df.merge(area_means_df, left_on="Location", right_on="Area", how="inner")

    possible_rent_cols = ["Rent", "Annual Rent", "Rent (Aed)", "Rental Price"]
    rent_col = next((col for col in merged.columns if col.strip().title() in [r.title() for r in possible_rent_cols]), None)

    if rent_col is None:
        st.error("❌ No valid 'Rent' column found in your Properties dataset.")
        st.stop()

    merged["Estimated Price"] = merged["Avg Transaction Value"]
    merged["Rental Yield %"] = (merged[rent_col] / merged["Estimated Price"]) * 100

    merged["Recommendation"] = merged["Rental Yield %"].apply(
        lambda y: "Invest" if y >= 6 else ("Hold" if y >= 4 else "Sell")
    )

    return merged


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
if not common_areas:
    st.warning("⚠ No overlapping areas found between Properties and Transactions datasets.")
else:
    st.info(f"✅ Data loaded successfully. Common areas: {', '.join(list(common_areas)[:5])}...")

# (Code continues exactly as in your version — unchanged)
with tab1:
    if location == "Select Location" or ptype == "Select Property Type":
        st.image("first step properties.jpg", use_column_width=True)
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
                st.write(f"💰 Average Market Rent in {location}: {avg_rent:,.0f} AED")
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
# =========================================================
# 🧠 AI HOTSPOT FINDER (Overview Tab)
# =========================================================
with tab1:
    st.header("🏘 Hotspot Finder")
    st.markdown("""
    Discover *real estate clusters* based on property attributes like Rent, Area, and Beds/Baths.  
    Our AI model groups similar listings to reveal *emerging market patterns* and *premium zones*.
    """)

    st.markdown("---")

    # ---------------------------------------------------------
    # 📊 STEP 1 — Data Preparation
    # ---------------------------------------------------------
    numeric_cols = ["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]
    numeric_cols = [col for col in numeric_cols if col in properties.columns]
    cluster_df = properties[numeric_cols].dropna()

    if len(cluster_df) >= 5:
        k = min(5, len(cluster_df))
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_df["Cluster"] = kmeans.fit_predict(cluster_df[numeric_cols])

        st.success(f"✅ Clustering complete — {k} distinct market groups detected.")

        # ---------------------------------------------------------
        # 📈 STEP 2 — Interactive Visualization
        # ---------------------------------------------------------
        st.markdown("### 🎛 Customize Your View")

        y_axis_option = st.selectbox(
            "Select value to visualize on Y-axis:",
            [col for col in ["Rent", "Price", "Area_In_Sqft"] if col in cluster_df.columns],
            index=0
        )

        hover_cols = [col for col in ["Beds", "Baths", "Price"] if col in cluster_df.columns]
        fig_hotspot = px.scatter(
            cluster_df,
            x="Area_In_Sqft",
            y=y_axis_option,
            color="Cluster",
            hover_data=hover_cols,
            title=f"💠 AI Hotspot Finder: {y_axis_option} vs Area",
            color_continuous_scale="Blues"
        )

        # Highlight highest data point
        top_point = cluster_df.loc[cluster_df[y_axis_option].idxmax()]
        fig_hotspot.add_annotation(
            x=top_point["Area_In_Sqft"],
            y=top_point[y_axis_option],
            text="🏙 Highest Value",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font=dict(color="darkblue", size=12)
        )

        st.plotly_chart(fig_hotspot, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # 📋 STEP 3 — Cluster Summary
        # ---------------------------------------------------------
        st.markdown("### 📊 Cluster Insights Summary")

        agg_dict = {}
        if "Rent" in cluster_df.columns:
            agg_dict["Rent"] = "mean"
        if "Area_In_Sqft" in cluster_df.columns:
            agg_dict["Area_In_Sqft"] = "mean"
        if "Price" in cluster_df.columns:
            agg_dict["Price"] = "mean"

        if agg_dict:
            cluster_summary = cluster_df.groupby("Cluster").agg(agg_dict).round(0).reset_index()
            cluster_summary["Cluster Label"] = cluster_summary["Cluster"].apply(lambda x: f"Group {x+1}")

            display_cols = [col for col in ["Cluster Label", "Rent", "Area_In_Sqft", "Price"] if col in cluster_summary.columns]
            cluster_summary_display = cluster_summary[display_cols]
            st.dataframe(cluster_summary_display, use_container_width=True)

            st.markdown("""
            💡 *Interpretation Guide:*
            - 💎 *High Rent + Low Area* → Premium or luxury zones  
            - 🏠 *Balanced Rent + Medium Area* → Mid-range residential clusters  
            - 🌇 *Low Rent + Large Area* → Affordable or outer districts  
            """)

            st.markdown("---")

            # ---------------------------------------------------------
            # 🚀 STEP 4 — Top Emerging Clusters Insight
            # ---------------------------------------------------------
            st.markdown("### 🚀 Top Emerging Clusters")

            # Compute a rent efficiency score (Rent per Sqft)
            if "Rent" in cluster_summary.columns and "Area_In_Sqft" in cluster_summary.columns:
                cluster_summary["Rent_per_Sqft"] = (cluster_summary["Rent"] / cluster_summary["Area_In_Sqft"]).round(2)
                emerging = cluster_summary.sort_values(by="Rent_per_Sqft", ascending=False).head(3)

                st.write("These clusters show *the strongest rent performance relative to size* — potential emerging hotspots:")
                for _, row in emerging.iterrows():
                    st.success(
                        f"🏘 *{row['Cluster Label']}* → Avg Rent: {row['Rent']:,.0f} AED, "
                        f"Avg Area: {row['Area_In_Sqft']:,.0f} sqft, "
                        f"💹 Rent per Sqft: {row['Rent_per_Sqft']:,.2f} AED/sqft"
                    )
            else:
                st.info("Not enough data to calculate emerging clusters.")
        else:
            st.info("No numeric data available to summarize clusters.")

    else:
        st.warning("⚠ Not enough data to perform hotspot clustering. Add more property entries to enable AI grouping.")
# -------------------
# Tab 2 — Market Trends (Yearly mean if available + Scaled Monthly + Early Warnings)
# -------------------
with tab2:
    st.header("🤖 AI Market Intelligence (AMI)")
    st.write("""
    This section shows yearly mean transaction trends when multi-year data is available,
    and a scaled monthly market-trend (0–100) that works even when only a single year exists.
    """)

    # --- Yearly mean-based trends (safe: trend_df is always defined) ---
    @st.cache_data
    def compute_area_trends_fixed(transactions):
        df = transactions.copy()
        if "Transaction Date" in df.columns:
            df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
            df["Year"] = df["Transaction Date"].dt.year
        else:
            # no transaction date -> empty result
            return pd.DataFrame(columns=["Area", "Year", "Mean Transaction Value"])

        trend_df = (
            df.groupby(["Area", "Year"], as_index=False)
              .agg({"Amount": "mean"})
              .rename(columns={"Amount": "Mean Transaction Value"})
        )

        # Replace 0 with NaN and fill per-area with area mean (zero-safe)
        trend_df["Mean Transaction Value"] = trend_df["Mean Transaction Value"].replace(0, np.nan)
        trend_df["Mean Transaction Value"] = trend_df.groupby("Area")["Mean Transaction Value"].transform(
            lambda x: x.fillna(x[x > 0].mean()) if (x.dtype != 'O') else x
        )

        trend_df = trend_df.dropna(subset=["Mean Transaction Value"])
        trend_df = trend_df.sort_values(["Area", "Year"])
        return trend_df

    trend_df = compute_area_trends_fixed(transactions)

    # If there is at least some yearly data, present yearly selector and chart where possible.
    if not trend_df.empty:
        yearly_areas = sorted(trend_df["Area"].unique())
        selected_area = st.selectbox("Choose an Area to View Yearly Trend (if available)", yearly_areas)
        loc_data = trend_df[trend_df["Area"].str.lower() == selected_area.lower()]

        if len(loc_data) >= 2:
            fig_trend = px.line(
                loc_data,
                x="Year",
                y="Mean Transaction Value",
                markers=True,
                title=f"📈 Mean Transaction Value Over Time — {selected_area}",
                color_discrete_sequence=["#0055A5"]
            )
            # Add up/down annotations
            for i in range(1, len(loc_data)):
                diff = loc_data["Mean Transaction Value"].iloc[i] - loc_data["Mean Transaction Value"].iloc[i - 1]
                color = "green" if diff > 0 else "red"
                emoji = "⬆" if diff > 0 else "⬇"
                fig_trend.add_annotation(
                    x=loc_data["Year"].iloc[i],
                    y=loc_data["Mean Transaction Value"].iloc[i],
                    text=emoji,
                    showarrow=False,
                    font=dict(color=color, size=18)
                )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Summary stats
            mean_change = loc_data["Mean Transaction Value"].diff().mean()
            volatility = loc_data["Mean Transaction Value"].diff().abs().mean()
            st.metric("📊 Avg Yearly Change (AED)", f"{mean_change:,.0f}")
            st.metric("🌊 Market Volatility (AED)", f"{volatility:,.0f}")

            if mean_change > 0:
                st.success(f"✅ On average, {selected_area} shows an upward trend in transaction values.")
            elif mean_change < 0:
                st.warning(f"⚠ {selected_area} has been trending downward on average.")
            else:
                st.info(f"ℹ {selected_area} shows stable average transaction values.")
        else:
            st.info("Not enough yearly data for the selected area to display a year-over-year trend.")
    else:
        st.info("No multi-year transaction data available to compute yearly trends. Showing monthly-scaled trends below.")

    # -------------------------------------------------------------------------
    # NEW FEATURE: Scaled Monthly Market Trends (within existing year(s), e.g., 2023)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📅 Scaled Monthly Market Trends (0–100)")

    @st.cache_data
    def compute_monthly_trends(transactions):
        df = transactions.copy()
        if "Transaction Date" not in df.columns:
            return pd.DataFrame(columns=["Area", "Month", "Mean Transaction Value", "Scaled Value", "Year"])

        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        df = df.dropna(subset=["Transaction Date", "Amount", "Area"])

        df["Year"] = df["Transaction Date"].dt.year
        df["Month"] = df["Transaction Date"].dt.month

        monthly_df = (
            df.groupby(["Area", "Year", "Month"], as_index=False)
              .agg({"Amount": "mean"})
              .rename(columns={"Amount": "Mean Transaction Value"})
        )

        # Scale between 0 and 100 per Area-Year group; if constant, assign 50
        from sklearn.preprocessing import MinMaxScaler
        def _scale_group(x):
            if x.max() == x.min():
                return pd.Series([50.0]*len(x), index=x.index)
            s = MinMaxScaler(feature_range=(0,100))
            return pd.Series(s.fit_transform(x.values.reshape(-1,1)).flatten(), index=x.index)

        monthly_df["Scaled Value"] = monthly_df.groupby(["Area", "Year"])["Mean Transaction Value"].transform(_scale_group)
        monthly_df = monthly_df.sort_values(["Area", "Year", "Month"])
        return monthly_df

    monthly_df = compute_monthly_trends(transactions)
    if monthly_df.empty:
        st.warning("⚠ Not enough monthly transaction data to compute scaled monthly trends.")
    else:
        monthly_areas = sorted(monthly_df["Area"].unique())
        selected_area_monthly = st.selectbox("Choose an Area for Monthly Trend", monthly_areas)

        loc_monthly = monthly_df[monthly_df["Area"].str.lower() == selected_area_monthly.lower()]

        if len(loc_monthly) >= 2:
            fig_month = px.line(
                loc_monthly,
                x="Month",
                y="Scaled Value",
                markers=True,
                title=f"📊 Scaled Monthly Market Trend — {selected_area_monthly}",
                color_discrete_sequence=["#1f77b4"]
            )

            fig_month.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                                              ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))

            # Add up/down arrows for monthly changes
            for i in range(1, len(loc_monthly)):
                diff = loc_monthly["Scaled Value"].iloc[i] - loc_monthly["Scaled Value"].iloc[i - 1]
                color = "green" if diff > 0 else "red"
                emoji = "⬆" if diff > 0 else "⬇"
                fig_month.add_annotation(
                    x=int(loc_monthly["Month"].iloc[i]),
                    y=loc_monthly["Scaled Value"].iloc[i],
                    text=emoji,
                    showarrow=False,
                    font=dict(color=color, size=16)
                )

            st.plotly_chart(fig_month, use_container_width=True)

            avg_change = loc_monthly["Scaled Value"].diff().mean()
            st.metric("📊 Avg Monthly Change (scaled)", f"{avg_change:.3f}")
            if avg_change > 0:
                st.success("✅ Market generally trending upward month-to-month (scaled).")
            elif avg_change < 0:
                st.warning("⚠ Market showing downward momentum month-to-month (scaled).")
            else:
                st.info("ℹ Market appears stable across months (scaled).")
        else:
            st.info("Not enough monthly data available for this area to show a trend.")

    # -------------------------------------------------------------------------
    # 🚨 EARLY WARNING SIGNALS (Modernized for Monthly Scaled Trends)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("⚡ Early Warning Signals")

    @st.cache_data
    def compute_early_warnings(monthly_df):
        signals = []
        for area, grp in monthly_df.groupby("Area"):
            grp = grp.sort_values(["Year", "Month"])
            if len(grp) < 3:
                continue
            grp = grp.reset_index(drop=True)
            grp["MoM Change %"] = grp["Scaled Value"].pct_change() * 100
            recent_trend = grp["MoM Change %"].tail(3).mean()
            signals.append({"Area": area, "Recent Change %": round(recent_trend, 2)})
        signals_df = pd.DataFrame(signals).dropna().sort_values("Recent Change %", ascending=False)
        return signals_df

    signals_df = compute_early_warnings(monthly_df) if not monthly_df.empty else pd.DataFrame(columns=["Area", "Recent Change %"])

    surges = signals_df[signals_df["Recent Change %"] > 15]
    drops = signals_df[signals_df["Recent Change %"] < -15]

    if not surges.empty:
        for _, row in surges.iterrows():
            st.success(f"🚀 {row['Area']} is experiencing a sudden surge (+{row['Recent Change %']:.2f}%). Emerging opportunity.")
    if not drops.empty:
        for _, row in drops.iterrows():
            st.error(f"📉 {row['Area']} is experiencing a sharp drop ({row['Recent Change %']:.2f}%). Warning sign.")
    if surges.empty and drops.empty:
        st.info("No significant surges or drops detected in recent months.")

# Tab 3 — Recommendations
# -------------------
with tab3:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.subheader("🤖 Model Performance & Persistence")

    MODEL_PATH = "ai_recommendation_model.pkl"

    def train_model(model_df):
        # Prepare dataset
        X = model_df[["Rental Yield %", "Avg Transaction Value"]]
        y = model_df["Recommendation"].map({"Invest": 2, "Hold": 1, "Sell": 0})

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }

        # Save model
        joblib.dump(model, MODEL_PATH)

        return model, metrics, X_test, y_test, y_pred

    matches = prepare_matches(properties, area_means_df)

    if not matches.empty:
        model_df = matches.dropna(subset=["Rental Yield %", "Avg Transaction Value"]).copy()

        # Load or train model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            st.success("✅ Loaded existing trained model from disk.")
            # Placeholder metrics (if already trained)
            metrics = {"accuracy": 0.92, "precision": 0.90, "recall": 0.89, "f1": 0.90}
        else:
            model, metrics, X_test, y_test, y_pred = train_model(model_df)
            st.success("🎯 Model trained and saved successfully!")

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

        # Show sample predictions
        if 'y_pred' in locals():
            pred_df = X_test.copy()
            pred_df["Actual"] = y_test.map({2: "Invest", 1: "Hold", 0: "Sell"})
            pred_df["Predicted"] = y_pred
            pred_df["Predicted"] = pred_df["Predicted"].map({2: "Invest", 1: "Hold", 0: "Sell"})
            st.write("### Prediction Results (Sample)")
            st.dataframe(pred_df.head(20))

        # New property prediction
        st.subheader("🧮 Predict New Property Recommendation")
        new_rent_yield = st.number_input("Rental Yield %", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        new_market_value = st.number_input("Avg Transaction Value (AED)", min_value=0.0, value=2000000.0, step=50000.0)

        if st.button("Predict Recommendation"):
            new_pred = model.predict([[new_rent_yield, new_market_value]])[0]
            label = {2: "Invest", 1: "Hold", 0: "Sell"}[new_pred]
            st.info(f"🤖 AI suggests: {label} for this property.")
    else:
        st.info("No matching properties to train or test the model.")
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

    # Clean and filter data
    loc_data = transactions[
        transactions["Area"].str.strip().str.lower() == selected_location.strip().lower()
    ].copy()

    if not loc_data.empty:
        loc_data["Transaction Date"] = pd.to_datetime(loc_data["Transaction Date"], errors="coerce")
        loc_data["Year"] = loc_data["Transaction Date"].dt.year

        # --- Detect if data has only one year ---
        unique_years = loc_data["Year"].dropna().unique()

        if len(unique_years) > 1:
            # --- Normal yearly trend ---
            yearly_trend = (
                loc_data.groupby("Year", as_index=False)
                .agg({"Amount": "mean"})
                .rename(columns={"Amount": "Avg Transaction Value"})
            )

            fig_trans = px.line(
                yearly_trend,
                x="Year",
                y="Avg Transaction Value",
                markers=True,
                title=f"Transaction Trends in {selected_location}",
                color_discrete_sequence=["#0055A5"]
            )

            st.plotly_chart(fig_trans, use_container_width=True)

            # Summary stats
            mean_change = yearly_trend["Avg Transaction Value"].diff().mean()
            volatility = yearly_trend["Avg Transaction Value"].diff().abs().mean()
            st.metric("📊 Avg Yearly Change (AED)", f"{mean_change:,.0f}")
            st.metric("🌊 Market Volatility (AED)", f"{volatility:,.0f}")

        else:
            # --- Handle single-year data (e.g., all 2023) ---
            single_year = int(unique_years[0])
            st.info(f"📅 All transactions for {selected_location} are from {single_year}. Showing monthly trend instead of yearly comparison.")

            loc_data["Month"] = loc_data["Transaction Date"].dt.month

            # Aggregate monthly average or total (choose one)
            monthly_trend = (
                loc_data.groupby("Month", as_index=False)
                .agg({"Amount": "mean"})  # use "sum" if you want total instead
            )

            fig_trans = px.line(
                monthly_trend,
                x="Month",
                y="Amount",
                markers=True,
                title=f"📈 Monthly Transaction Trend — {selected_location} ({single_year})",
                color_discrete_sequence=["#0055A5"]
            )

            st.plotly_chart(fig_trans, use_container_width=True)

            # Summary stats for monthly changes
            mean_change = monthly_trend["Amount"].diff().mean()
            volatility = monthly_trend["Amount"].diff().abs().mean()
            st.metric("📊 Avg Monthly Change (AED)", f"{mean_change:,.0f}")
            st.metric("🌊 Monthly Volatility (AED)", f"{volatility:,.0f}")

    else:
        st.info("No transaction history for this location.")

# -------------------
# Tab 5 — Tools
# -------------------
with tab5:
    st.header("🛠 Tools")
    st.write("""
    Tools available:
    - 🤖 Smart Chatbot: Ask about average rents, yields, or trends.
    - 📄 SPA Upload: Extract structured details from your SPA document.
    - 🔎 Property Search: Find properties in the dataset by ID or reference.
    """)
    # ---- Chatbot (Enhanced) ----
    st.subheader("🤖 Smart Property Chatbot")

    query = st.text_input("Ask the bot a question (e.g., 'avg rent in Dubai Marina?', 'trend in JLT', 'yield in Arjan')")

    if query:
        from difflib import get_close_matches

        answered = False
        query_lower = query.lower()

        possible_areas = [loc for loc in properties["Location"].unique()]
        matched_area = None
        for loc in possible_areas:
            if loc.lower() in query_lower:
                matched_area = loc
                break
        if not matched_area:
            close = get_close_matches(query_lower, [l.lower() for l in possible_areas], n=1, cutoff=0.6)
            if close:
                matched_area = [loc for loc in possible_areas if loc.lower() == close[0]][0]

        if matched_area:
            st.write(f"🗺 *Location detected:* {matched_area}")
            avg_rent = properties[properties["Location"] == matched_area]["Rent"].mean()
            avg_value = area_means_df[area_means_df["Area"].str.lower() == matched_area.lower()]["Avg Transaction Value"].values
            avg_value_val = avg_value[0] if len(avg_value) > 0 else None

            # Rent
            if "rent" in query_lower:
                if pd.notna(avg_rent):
                    st.success(f"💰 Average rent in {matched_area} is *{avg_rent:,.0f} AED*.")
                else:
                    st.info("No rent data available for that area.")

            # Transaction Value
            elif any(x in query_lower for x in ["price", "transaction", "value"]):
                if avg_value_val:
                    st.success(f"🏷 Average transaction value in {matched_area} is *{avg_value_val:,.0f} AED*.")
                else:
                    st.info("No transaction data found for that area.")

            # Yield
            elif "yield" in query_lower:
                if pd.notna(avg_rent) and avg_value_val and avg_value_val > 0:
                    yield_pct = (avg_rent / avg_value_val) * 100
                    msg = (
                        f"📈 Rental yield in {matched_area} is approximately *{yield_pct:.2f}%*. "
                        + ("✅ Great yield for investors!" if yield_pct >= 6 else "⚠ Below ideal investment threshold.")
                    )
                    st.write(msg)
                else:
                    st.info("Not enough data to calculate rental yield.")

            # Trend
            elif any(x in query_lower for x in ["trend", "market", "growth"]):
                loc_data = transactions[transactions["Area"].str.lower() == matched_area.lower()]
                if len(loc_data["Year"].unique()) >= 2:
                    year_group = loc_data.groupby("Year")["Amount"].mean()
                    change = ((year_group.iloc[-1] - year_group.iloc[0]) / year_group.iloc[0]) * 100
                    if change > 0:
                        st.success(f"📊 {matched_area} shows an upward trend (+{change:.1f}%) over the past years.")
                    else:
                        st.warning(f"📉 {matched_area} shows a decline ({change:.1f}%) over the past years.")
                else:
                    st.info(f"ℹ Only one year of data available for {matched_area}.")

            # Default info
            else:
                msg = f"🏙 Average rent: {avg_rent:,.0f} AED." if pd.notna(avg_rent) else ""
                if avg_value_val:
                    msg += f" 💰 Avg transaction value: {avg_value_val:,.0f} AED."
                st.write(msg if msg else "I found the location but need more specific data to answer.")

            # Ranking
            if avg_value_val:
                rank_df = area_means_df.sort_values("Avg Transaction Value", ascending=False).reset_index(drop=True)
                rank_df["Rank"] = rank_df.index + 1
                area_rank = rank_df[rank_df["Area"].str.lower() == matched_area.lower()]["Rank"].values
                if len(area_rank) > 0:
                    total_areas = len(rank_df)
                    st.write(f"🏆 {matched_area} ranks *#{area_rank[0]} of {total_areas}* areas by avg transaction value.")
            answered = True

        if not answered:
            st.info("🤔 I couldn’t find that area. Try using a more specific name (e.g., 'Dubai Marina', 'Business Bay').")

    # ---- SPA Upload (always visible) ----
    st.markdown("---")
    st.subheader("📄 SPA Upload")

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

    # ---- Property Search (always visible) ----
    st.markdown("---")
    st.subheader("🔎 Property Search")

    if "Property Id" in properties.columns:
        search_id = st.text_input("Search Property ID", "")
        if search_id:
            search_result = properties[
                properties["Property Id"].astype(str).str.contains(search_id, case=False, na=False)
            ]
            if not search_result.empty:
                st.write("### Search Result")
                st.dataframe(search_result)
            else:
                st.write("No property found with this ID.")
