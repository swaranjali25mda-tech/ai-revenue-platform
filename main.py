from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from ml_models import train_models
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI()
latest_uploaded_file = None

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):

    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    global latest_uploaded_file
    latest_uploaded_file = file_location
        

    # Read CSV
    df = pd.read_csv(file_location)

    # Calculate KPIs
    total_revenue = df["revenue"].sum()
    avg_revenue = df["revenue"].mean()
    max_revenue = df["revenue"].max()

    # Generate Graph
    plt.figure()
    df["date"] = pd.to_datetime(df["date"])
    plt.plot(df["date"], df["revenue"])
    plt.xticks(rotation=45)
    plt.title("Revenue Trend")
    plt.tight_layout()
    plt.savefig("static/images/graph.png")
    plt.close()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total": round(total_revenue, 2),
        "average": round(avg_revenue, 2),
        "max": round(max_revenue, 2)
    })

@app.get("/forecast", response_class=HTMLResponse)
def forecast(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("forecast.html", {
            "request": request,
            "error": "Please upload a file first."
        })

    df = pd.read_csv(latest_uploaded_file)

    df = df.rename(columns={"date": "ds", "revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )

    df["ds"] = pd.to_datetime(df["ds"])

# Remove duplicate dates
df = df.drop_duplicates(subset=["ds"])

# Sort by date
df = df.sort_values("ds")

# Reset index
df = df.reset_index(drop=True)

    model.fit(df)

    future = model.make_future_dataframe(periods=3, freq='ME')
    forecast = model.predict(future)
    # Save full forecast to CSV
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_output.to_csv("data/forecast_output.csv", index=False)

    # Take only future predictions (last 3 rows)
    forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3)

    # Convert to readable format
    forecast_data = []
    for _, row in forecast_table.iterrows():
        forecast_data.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "prediction": round(row["yhat"], 2),
            "lower": round(row["yhat_lower"], 2),
            "upper": round(row["yhat_upper"], 2)
        })

    fig = model.plot(forecast)
    fig.savefig("static/images/forecast.png")
    plt.close(fig)

    return templates.TemplateResponse("forecast.html", {
    "request": request,
    "forecast_data": forecast_data
})

from fastapi.responses import FileResponse

@app.get("/download-forecast")
def download_forecast():
    file_path = "data/forecast_output.csv"
    return FileResponse(
        path=file_path,
        filename="forecast_results.csv",
        media_type="text/csv"
    )

@app.get("/ml-models", response_class=HTMLResponse)
def ml_models_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("ml_models.html", {
            "request": request,
            "error": "Please upload a file first."
        })

    results = train_models(latest_uploaded_file)

    return templates.TemplateResponse("ml_models.html", {
        "request": request,
        "results": results
    })


@app.get("/segmentation", response_class=HTMLResponse)
def segmentation_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("segmentation.html", {
            "request": request,
            "error": "Upload file first."
        })

    df = pd.read_csv(latest_uploaded_file)

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["revenue"]])

    # Create Graph
    plt.figure()
    plt.scatter(df.index, df["revenue"], c=df["cluster"])
    plt.title("Customer Segmentation")
    plt.xlabel("Index")
    plt.ylabel("Revenue")
    plt.savefig("static/images/cluster.png")
    plt.close()

    # Cluster Summary (Clean Version)
    cluster_summary = df.groupby("cluster").agg(
        Avg_Revenue=("revenue", "mean"),
        Min_Revenue=("revenue", "min"),
        Max_Revenue=("revenue", "max"),
        Customer_Count=("revenue", "count")
    ).reset_index()

    # Add Business Label
    def label_cluster(avg):
        if avg > df["revenue"].mean():
            return "High Value"
        else:
            return "Low / Medium Value"

    cluster_summary["Segment_Type"] = cluster_summary["Avg_Revenue"].apply(label_cluster)

    cluster_data = cluster_summary.to_dict(orient="records")

    # Create Pie Chart for Cluster Distribution
    cluster_counts = df["cluster"].value_counts()

    plt.figure()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title("Cluster Distribution")
    plt.savefig("static/images/cluster_pie.png")
    plt.close()

    # Business Insight
    highest_cluster = cluster_summary.sort_values(by="Avg_Revenue", ascending=False).iloc[0]

    insight_text = f"""
    Cluster {int(highest_cluster['cluster'])} represents the highest value customers.
    They generate an average revenue of ₹{round(highest_cluster['Avg_Revenue'],2)}. 
    Business should target them with loyalty programs and premium offers.
    """

    return templates.TemplateResponse("segmentation.html", {
    "request": request,
    "cluster_data": cluster_data,
    "insight": insight_text
})

@app.get("/anomaly", response_class=HTMLResponse)
def anomaly_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("anomaly.html", {
            "request": request,
            "error": "Please upload file first."
        })

    df = pd.read_csv(latest_uploaded_file)

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    features = df[["revenue", "month", "year"]]

    model = IsolationForest(contamination=0.1)
    df["anomaly"] = model.fit_predict(features)

    # -1 = anomaly, 1 = normal
    anomaly_count = len(df[df["anomaly"] == -1])

    # Create Graph
    plt.figure(figsize=(8,5))

    normal = df[df["anomaly"] == 1]
    abnormal = df[df["anomaly"] == -1]

    plt.scatter(normal["date"], normal["revenue"], label="Normal")
    plt.scatter(abnormal["date"], abnormal["revenue"], label="Anomaly")

    plt.legend()
    plt.title("Revenue Anomaly Detection")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/anomaly.png")
    plt.close()

    anomalies = df[df["anomaly"] == -1][["date", "revenue"]]

    return templates.TemplateResponse("anomaly.html", {
        "request": request,
        "anomaly_count": anomaly_count,
        "anomalies": anomalies.to_dict(orient="records")
    })

@app.get("/learn-more", response_class=HTMLResponse)
def learn_more_page(request: Request):
    return templates.TemplateResponse("learn_more.html", {"request": request})

@app.get("/summary", response_class=HTMLResponse)
def summary_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("summary.html", {
            "request": request,
            "error": "Upload file first."
        })

    df = pd.read_csv(latest_uploaded_file)
    df["date"] = pd.to_datetime(df["date"])

    total = df["revenue"].sum()
    avg = df["revenue"].mean()
    max_value = df["revenue"].max()
    min_value = df["revenue"].min()

    # Monthly Growth
    df = df.sort_values("date")
    df["growth"] = df["revenue"].pct_change() * 100
    avg_growth = df["growth"].mean()

    best_month = df.loc[df["revenue"].idxmax(), "date"].strftime("%B %Y")
    worst_month = df.loc[df["revenue"].idxmin(), "date"].strftime("%B %Y")

    trend = "Upward 📈" if avg_growth > 0 else "Downward 📉"

    return templates.TemplateResponse("summary.html", {
        "request": request,
        "total": round(total, 2),
        "average": round(avg, 2),
        "max": round(max_value, 2),
        "min": round(min_value, 2),
        "avg_growth": round(avg_growth, 2),
        "best_month": best_month,
        "worst_month": worst_month,
        "trend": trend
    })

@app.get("/download-report")
def download_report():

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return {"error": "Upload file first."}

    return FileResponse(
        path=latest_uploaded_file,
        filename="revenue_report.csv",
        media_type="text/csv"
    )

@app.get("/generate-pdf")
def generate_pdf():

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return {"error": "Upload file first."}

    df = pd.read_csv(latest_uploaded_file)

    total = df["revenue"].sum()
    avg = df["revenue"].mean()
    max_value = df["revenue"].max()

    pdf_path = "static/executive_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Revenue Executive Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Total Revenue: ₹{round(total,2)}", styles["Normal"]))
    elements.append(Paragraph(f"Average Revenue: ₹{round(avg,2)}", styles["Normal"]))
    elements.append(Paragraph(f"Highest Revenue: ₹{round(max_value,2)}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Business Insight
    if df["revenue"].iloc[-1] > avg:
        insight = "Revenue trend is positive. Business growth detected."
    else:
        insight = "Revenue below average. Strategic optimization recommended."

    elements.append(Paragraph("Executive Insight:", styles["Heading2"]))
    elements.append(Paragraph(insight, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Add chart image if exists
    from reportlab.platypus import Image
    if os.path.exists("static/images/graph.png"):
        elements.append(Paragraph("Revenue Trend Chart:", styles["Heading2"]))
        elements.append(Image("static/images/graph.png", width=400, height=250))

    doc.build(elements)

    return FileResponse(
        path=pdf_path,
        filename="AI_Executive_Report.pdf",
        media_type="application/pdf"
    )

@app.get("/model-performance", response_class=HTMLResponse)
def model_performance_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("model_performance.html", {
            "request": request,
            "error": "Upload file first."
        })

    df = pd.read_csv(latest_uploaded_file)
    df = df.rename(columns={"date": "ds", "revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    actual = df["y"]
    predicted = forecast["yhat"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # Plot Actual vs Predicted
    plt.figure(figsize=(8,5))
    plt.plot(df["ds"], actual, label="Actual")
    plt.plot(df["ds"], predicted, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/model_performance.png")
    plt.close()

    return templates.TemplateResponse("model_performance.html", {
        "request": request,
        "mae": round(mae,2),
        "rmse": round(rmse,2),
        "r2": round(r2,2)
    })

@app.get("/ai-insights", response_class=HTMLResponse)
def ai_insights_page(request: Request):

    global latest_uploaded_file

    if latest_uploaded_file is None:
        return templates.TemplateResponse("ai_insights.html", {
            "request": request,
            "error": "Upload file first."
        })

    df = pd.read_csv(latest_uploaded_file)

    mean_rev = df["revenue"].mean()
    last_rev = df["revenue"].iloc[-1]
    volatility = df["revenue"].std()

    insights = []

    if last_rev > mean_rev:
        insights.append("Revenue is currently above historical average.")
    else:
        insights.append("Revenue is below historical average.")

    if volatility > mean_rev * 0.2:
        insights.append("Revenue shows high volatility. Risk management recommended.")
    else:
        insights.append("Revenue trend is stable.")

    top_month = df.loc[df["revenue"].idxmax()]
    insights.append(f"Highest revenue recorded on {top_month['date']}.")

    return templates.TemplateResponse("ai_insights.html", {
        "request": request,
        "insights": insights
    })
