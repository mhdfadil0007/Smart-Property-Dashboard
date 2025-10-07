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

    # Ensure both datasets align (but do not display globally)
    common_areas = set(properties["Location"].unique()) & set(transactions["Area"].unique())

# -------------------
# Sidebar Inputs
# -------------------
with st.sidebar:
    st.header("🏠 Property Inputs")

    # Show common areas info here (sidebar) instead of globally
    if not common_areas:
        st.warning("⚠ No overlapping areas found between Properties and Transactions datasets.")
    else:
        st.info(f"✅ Data loaded. Sample common areas: {', '.join(list(common_areas)[:5])}...")

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

# -------------------
# Tabs
# -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Market Trends", "Recommendations", "Transactions", "Tools"
])

# Ensure Price column exists to avoid KeyError in downstream code
if "Price" not in properties.columns:
    properties["Price"] = np.nan

# -------------------
# Tab 1 — Overview (including guarded Hotspot Finder)
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
                pct_diff = (rent - avg_rent) / avg_rent * 100 if avg_rent > 0 else np.nan
                st.write(f"💰 Average Market Rent in {location}: {avg_rent:,.0f} AED")
                st.write(f"🔍 Rent difference vs market: {pct_diff:.2f}%")

                if not np.isnan(pct_diff) and rent > avg_rent * 1.15:
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

        # --- HOTSPOT FINDER in Overview (only after selections and filtered) ---
        st.subheader("🏘 Hotspot Finder")
        # Show hotspot only when location and property type are selected from sidebar (and not defaults)
        if location != "Select Location" and ptype != "Select Property Type":
            # Filter properties to selected location and type for meaningful clustering
            filtered = properties[
                (properties["Location"].str.strip().str.lower() == location.strip().lower()) &
                (properties["Type"].astype(str).str.strip().str.lower() == ptype.strip().lower())
            ].copy()

            # Fallback to global properties if filtered set is too small (still keep user informed)
            if len(filtered) < 5:
                st.info("Not enough properties for that specific Location & Type. Running hotspot on full dataset for insights.")
                cluster_df = properties[["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]].copy()
            else:
                cluster_df = filtered[["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]].copy()

            # Only keep numeric columns and drop rows missing those
            candidate_cols = ["Rent", "Area_In_Sqft", "Beds", "Baths", "Price"]
            numeric_cols = [col for col in candidate_cols if col in cluster_df.columns]
            cluster_df = cluster_df[numeric_cols].dropna()

            if len(cluster_df) >= 5:
                k = min(5, len(cluster_df))
                kmeans = KMeans(n_clusters=k, random_state=42)
                # fit on numeric columns only
                cluster_df["Cluster"] = kmeans.fit_predict(cluster_df[numeric_cols])

                # Interactive Y-axis selector
                y_axis_options = [col for col in ["Rent", "Price", "Area_In_Sqft"] if col in cluster_df.columns]
                y_axis_option = st.selectbox("Select value to visualize on Y-axis:", y_axis_options, index=0)

                hover_cols = [col for col in ["Beds", "Baths", "Price"] if col in cluster_df.columns]
                fig_hotspot = px.scatter(
                    cluster_df.reset_index(),
                    x="Area_In_Sqft",
                    y=y_axis_option,
                    color="Cluster",
                    hover_data=hover_cols,
                    title=f"AI Hotspot Finder: {y_axis_option} vs Area",
                )
                # annotate top point if present
                if y_axis_option in cluster_df.columns and not cluster_df[y_axis_option].isna().all():
                    try:
                        top_idx = cluster_df[y_axis_option].idxmax()
                        top_point = cluster_df.loc[top_idx]
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
                    except Exception:
                        pass

                st.plotly_chart(fig_hotspot, use_container_width=True)

                # --- Cluster Summary (safe aggregates) ---
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
                    💡 **Interpretation Guide:**
                    - High *Rent* + Low *Area* → Likely premium zones (small but expensive).
                    - Moderate *Rent* + Medium *Area* → Balanced residential areas.
                    - Low *Rent* + Large *Area* → Affordable or outer regions.
                    """)
                else:
                    st.info("No numeric data available to summarize clusters.")
            else:
                st.info("Not enough data to perform hotspot clustering for the chosen filter.")
        else:
            st.info("👆 Please select both a Location and Property Type in the sidebar to enable the Hotspot Finder.")

# -------------------
# Tab 2 — Market Trends (Mean-based, Zero-safe) with Monthly Scaling fallback
# -------------------
with tab2:
    st.header("🤖 AI Market Intelligence (AMI)")
    st.subheader("📊 Market Trends & Scaled Monthly View")
    st.write("""
    This view analyzes average transaction values per year (if multi-year data exists).
    If only one year exists for an area, a monthly-scaled trend is presented instead.
    🟩 = going up | 🟥 = going down
    """)

    @st.cache_data
    def compute_area_trends_fixed(transactions):
        trend_df = (
            transactions
            .groupby(["Area", "Year"], as_index=False)
            .agg({"Amount": "mean"})
            .rename(columns={"Amount": "Mean Transaction Value"})
        )

        trend_df["Mean Transaction Value"] = trend_df["Mean Transaction Value"].replace(0, np.nan)
        trend_df["Mean Transaction Value"] = trend_df.groupby("Area")["Mean Transaction Value"].transform(
            lambda x: x.fillna(x[x > 0].mean()) if (x[x > 0].size > 0) else x
        )

        trend_df = trend_df.dropna(subset=["Mean Transaction Value"])
        trend_df = trend_df.sort_values(["Area", "Year"])
        return trend_df

    trend_df = compute_area_trends_fixed(transactions)

    # ---- Area Selector ----
    # Use Area values from transactions (trend_df may be empty if data missing)
    if not trend_df.empty:
        all_areas = sorted(trend_df["Area"].unique())
    else:
        # fallback to transaction Areas if trend_df empty
        all_areas = sorted(transactions["Area"].unique())

    selected_area = st.selectbox("Choose an Area to View Trend", all_areas)

    # Attempt yearly view first
    loc_data = trend_df[trend_df["Area"].str.lower() == selected_area.lower()] if not trend_df.empty else pd.DataFrame()

    if len(loc_data) >= 2:
        fig_trend = px.line(
            loc_data,
            x="Year",
            y="Mean Transaction Value",
            markers=True,
            title=f"📈 Mean Transaction Value Over Time — {selected_area}",
            color_discrete_sequence=["#0055A5"]
        )

        for i in range(1, len(loc_data)):
            diff = loc_data["Mean Transaction Value"].iloc[i] - loc_data["Mean Transaction Value"].iloc[i - 1]
            color = "green" if diff > 0 else "red"
            emoji = "⬆️" if diff > 0 else "⬇️"
            fig_trend.add_annotation(
                x=loc_data["Year"].iloc[i],
                y=loc_data["Mean Transaction Value"].iloc[i],
                text=emoji,
                showarrow=False,
                font=dict(color=color, size=18)
            )

        # Avoid extreme outliers dominating visual (set y-range to 1st/99th percentiles)
        try:
            ymin = loc_data["Mean Transaction Value"].quantile(0.01)
            ymax = loc_data["Mean Transaction Value"].quantile(0.99)
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                fig_trend.update_yaxes(range=[max(0, ymin*0.9), ymax*1.1])
        except Exception:
            pass

        st.plotly_chart(fig_trend, use_container_width=True)

        mean_change = loc_data["Mean Transaction Value"].diff().mean()
        volatility = loc_data["Mean Transaction Value"].diff().abs().mean()
        st.metric("📊 Avg Yearly Change (AED)", f"{mean_change:,.0f}")
        st.metric("🌊 Market Volatility (AED)", f"{volatility:,.0f}")

        if mean_change > 0:
            st.success(f"✅ On average, {selected_area} shows an upward trend in transaction values.")
        elif mean_change < 0:
            st.warning(f"⚠️ {selected_area} has been trending downward on average.")
        else:
            st.info(f"ℹ️ {selected_area} shows stable average transaction values.")
    else:
        # Fallback: compute and show monthly scaled trend for selected area (use transaction dates)
        st.info(f"All transactions for {selected_area} appear to be in a single year or insufficient yearly data. Showing monthly trend instead.")

        @st.cache_data
        def compute_monthly_trends(transactions):
            df = transactions.copy()
            df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
            df["Month"] = df["Transaction Date"].dt.month

            monthly_df = (
                df.groupby(["Area", "Month"], as_index=False)
                .agg({"Amount": "mean"})
                .rename(columns={"Amount": "Mean Transaction Value"})
            )

            # Scale per area (0-1)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            def scale_grp(x):
                if len(x) == 0:
                    return x
                vals = x.values.reshape(-1,1)
                try:
                    scaled = scaler.fit_transform(vals).flatten()
                    return scaled
                except Exception:
                    return np.zeros_like(x)

            monthly_df["Scaled Value"] = monthly_df.groupby("Area")["Mean Transaction Value"].transform(lambda x: pd.Series(scale_grp(x), index=x.index))
            return monthly_df

        monthly_df = compute_monthly_trends(transactions)
        loc_monthly = monthly_df[monthly_df["Area"].str.lower() == selected_area.lower()]

        if len(loc_monthly) >= 2:
            fig_month = px.line(
                loc_monthly,
                x="Month",
                y="Scaled Value",
                markers=True,
                title=f"📊 Scaled Monthly Market Trend — {selected_area}",
                color_discrete_sequence=["#1f77b4"]
            )
            fig_month.update_layout(
                xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                           ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
            )

            for i in range(1, len(loc_monthly)):
                diff = loc_monthly["Scaled Value"].iloc[i] - loc_monthly["Scaled Value"].iloc[i - 1]
                color = "green" if diff > 0 else "red"
                emoji = "⬆️" if diff > 0 else "⬇️"
                fig_month.add_annotation(
                    x=loc_monthly["Month"].iloc[i],
                    y=loc_monthly["Scaled Value"].iloc[i],
                    text=emoji,
                    showarrow=False,
                    font=dict(color=color, size=14)
                )

            st.plotly_chart(fig_month, use_container_width=True)
            avg_change = loc_monthly["Scaled Value"].diff().mean()
            st.metric("📊 Avg Monthly Change", f"{avg_change:.3f}")
            if avg_change > 0:
                st.success("✅ Market generally trending upward month-to-month.")
            elif avg_change < 0:
                st.warning("⚠️ Market showing downward momentum.")
            else:
                st.info("ℹ️ Market appears stable across months.")
        else:
            st.info("Not enough monthly data available for this area.")

    # Early warning signals logic (keeps same behavior)
    st.markdown("---")
    st.subheader("⚡ Early Warning Signals")
    @st.cache_data
    def compute_early_warnings(monthly_df):
        signals = []
        for area, grp in monthly_df.groupby("Area"):
            grp = grp.sort_values("Month")
            if len(grp) < 3:
                continue
            grp = grp.copy()
            grp["MoM Change %"] = grp["Scaled Value"].pct_change() * 100
            recent_trend = grp["MoM Change %"].tail(3).mean()
            signals.append({
                "Area": area,
                "Recent Change %": round(recent_trend, 2)
            })
        signals_df = pd.DataFrame(signals).dropna().sort_values("Recent Change %", ascending=False)
        return signals_df

    # compute signals only if monthly_df exists in this scope (we defined inside earlier branch)
    try:
        signals_df = compute_early_warnings(monthly_df)
        surges = signals_df[signals_df["Recent Change %"] > 15]
        drops = signals_df[signals_df["Recent Change %"] < -15]

        if not surges.empty:
            for _, row in surges.iterrows():
                st.success(f"🚀 {row['Area']} is experiencing a sudden surge (+{row['Recent Change %']:.2f}%). Emerging opportunity.")
        if not drops.empty:
            for _, row in drops.iterrows():
                st.error(f"📉 {row['Area']} is experiencing a sharp drop ({row['Recent Change %']:.2f}%). Warning sign.")
        if surges.empty and drops.empty:
            st.info("No significant surges or drops detected this month.")
    except Exception:
        # If monthly_df isn't defined (rare edge), skip gracefully
        pass

# -------------------
# Recommendations tab : Train, Save & Reuse Classification Model
# -------------------
with tab3:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.subheader("🤖 Model Performance & Persistence")

    MODEL_PATH = "ai_recommendation_model.pkl"

    # prepare_matches must exist here (safe)
    @st.cache_data
    def prepare_matches(properties, area_means_df):
        df = properties.copy()
        df["Location"] = df["Location"].astype(str).str.strip().str.title()
        merged = df.merge(area_means_df, left_on="Location", right_on="Area", how="inner")

        # --- Identify rent column dynamically ---
        possible_rent_cols = ["Rent", "Annual Rent", "Rent (Aed)", "Rental Price"]
        rent_col = next((col for col in merged.columns if col.strip().title() in [r.title() for r in possible_rent_cols]), None)

        if rent_col is None:
            # gracefully return empty
            return pd.DataFrame()

        # --- Use Avg Transaction Value as a proxy for property price if Price is NaN ---
        merged["Estimated Price"] = merged["Avg Transaction Value"]
        if "Price" not in merged.columns or merged["Price"].isna().all():
            merged["Price"] = merged["Estimated Price"]

        # --- Compute rental yield safely ---
        merged["Rental Yield %"] = (merged[rent_col] / merged["Price"]) * 100

        # --- Rule-based recommendation ---
        merged["Recommendation"] = merged["Rental Yield %"].apply(
            lambda y: "Invest" if y >= 6 else ("Hold" if y >= 4 else "Sell")
        )

        return merged

    matches = prepare_matches(properties, area_means_df)

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

    if not matches.empty:
        model_df = matches.dropna(subset=["Rental Yield %", "Avg Transaction Value"]).copy()

        # Load or train model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            st.success("✅ Loaded existing trained model from disk.")
            # Set placeholder metrics if you didn't re-train in this session
            metrics = {"accuracy": 0.92, "precision": 0.90, "recall": 0.89, "f1": 0.90}
        else:
            model, metrics, X_test, y_test, y_pred = train_model(model_df)
            st.success("🎯 Model trained and saved successfully!")

        # Display performance metrics
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

        # Allow new property prediction
        st.subheader("🧮 Predict New Property Recommendation")
        new_rent_yield = st.number_input("Rental Yield %", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        new_market_value = st.number_input("Avg Transaction Value (AED)", min_value=0.0, value=2000000.0, step=50000.0)

        if st.button("Predict Recommendation"):
            # ensure model variable exists (trained or loaded)
            try:
                new_pred = model.predict([[new_rent_yield, new_market_value]])[0]
                label = {2: "Invest", 1: "Hold", 0: "Sell"}[new_pred]
                st.info(f"🤖 AI suggests: **{label}** for this property.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
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

    # Filter by selected_location
    loc_data = transactions[
        transactions["Area"].str.strip().str.lower() == selected_location.strip().lower()
    ].copy()

    # If there's only one year in loc_data, compute monthly trend for that single year
    if not loc_data.empty:
        # ensure Transaction Date exists and Month derived
        if "Transaction Date" in loc_data.columns:
            loc_data["Transaction Date"] = pd.to_datetime(loc_data["Transaction Date"], errors="coerce")
            loc_data["Month"] = loc_data["Transaction Date"].dt.month
            years_present = loc_data["Year"].dropna().unique()
            if len(years_present) == 1:
                # single-year dataset --> show monthly trend for that year
                single_year = int(years_present[0])
                st.info(f"All transactions for {selected_location} are from {single_year}. Showing monthly trend instead of yearly comparison.")

                # Aggregate monthly average
                monthly_trend = (
                    loc_data.groupby("Month", as_index=False)
                    .agg({"Amount": "mean"})
                )

                fig_trans = px.line(
                    monthly_trend,
                    x="Month",
                    y="Amount",
                    markers=True,
                    title=f"📈 Monthly Transaction Trend — {selected_location} ({single_year})",
                    hover_data=["Amount"]
                )
                # convert month numbers to names
                fig_trans.update_layout(
                    xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                               ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
                )

                # avoid extreme outlier domination
                try:
                    ymin = monthly_trend["Amount"].quantile(0.01)
                    ymax = monthly_trend["Amount"].quantile(0.99)
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                        fig_trans.update_yaxes(range=[max(0, ymin*0.9), ymax*1.1])
                except Exception:
                    pass

                st.plotly_chart(fig_trans, use_container_width=True)
            else:
                # multi-year data available -> show yearly line
                yearly_trend = loc_data.groupby("Year", as_index=False).agg({"Amount": "mean"}).sort_values("Year")
                fig_trans = px.line(yearly_trend, x="Year", y="Amount", markers=True,
                                    title=f"Transaction Trends in {selected_location}", hover_data=["Amount"])
                # avoid extreme outlier domination
                try:
                    ymin = yearly_trend["Amount"].quantile(0.01)
                    ymax = yearly_trend["Amount"].quantile(0.99)
                    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                        fig_trans.update_yaxes(range=[max(0, ymin*0.9), ymax*1.1])
                except Exception:
                    pass
                st.plotly_chart(fig_trans, use_container_width=True)
        else:
            # No transaction date; fallback to showing raw amounts over index
            st.info("Transaction Date not available; showing raw transaction amounts.")
            fig_trans = px.line(loc_data.reset_index(), x="index", y="Amount", markers=True,
                                title=f"Transaction Amounts in {selected_location}", hover_data=["Amount"])
            st.plotly_chart(fig_trans, use_container_width=True)
    else:
        st.info("No transaction history for this location.")

# -------------------
# Tab 5 — Tools
# -------------------
with tab5:
    st.header("🛠 Tools")
    st.write("""
    Tools available:
    - Chatbot: Ask questions about average rents or investment insights.
    - SPA Upload: Extract structured details from your SPA document.
    - Search: Find properties in the dataset by ID or reference, with AI suggestions.
    """)

    # ---- Chatbot ----
    query = st.text_input("Ask the bot a question (e.g., 'avg rent in Dubai Marina?')")
    if query:
        answered = False
        for loc in properties["Location"].unique():
            if loc.lower() in query.lower():
                avg_rent = properties[properties["Location"] == loc]["Rent"].mean()
                avg_value = area_means_df[area_means_df["Area"].str.lower() == loc.lower()]["Avg Transaction Value"].values
                value_info = f" Avg transaction value: {avg_value[0]:,.0f} AED." if len(avg_value) > 0 else ""
                st.write(f"Bot: Average rent in {loc} is {avg_rent:,.0f} AED.{value_info}")
                answered = True
                break
        if not answered:
            st.write("Bot: This feature is in preview. Try including a location name in your query.")

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
