import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from flask import Flask, request, render_template_string
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io
import base64
import os

app = Flask(__name__)

def forecast_sales(df):

    sales = df["sales"].values
    days = np.arange(len(sales)).reshape(-1,1)

    model = XGBRegressor(n_estimators=200)
    model.fit(days,sales)

    predictions = model.predict(days)
    accuracy = r2_score(sales,predictions)

    future_days = np.arange(len(sales),len(sales)+30).reshape(-1,1)
    forecast = model.predict(future_days)

    return forecast,accuracy,predictions


def inventory_analysis(stock,forecast):

    future_demand = int(np.sum(forecast))

    if stock < future_demand:

        reorder_qty = future_demand - stock
        over_stock = 0
        status = "⚠ Reorder Required"

    else:

        reorder_qty = 0
        over_stock = stock - future_demand
        status = "✅ Stock OK"

    return status,reorder_qty,over_stock,future_demand


html = """

<!DOCTYPE html>
<html>
<head>

<title>AI Inventory Intelligence Dashboard</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

</head>

<body style="background:#eef2f7">

<div class="container mt-5">

<h2 class="text-center">AI Smart Inventory Forecast System</h2>

<div class="card p-4 mt-4">

<form method="POST" enctype="multipart/form-data">

<label>Upload CSV</label>

<input type="file" name="file" class="form-control">

<br>

<label>Current Stock</label>

<input type="number" name="stock" class="form-control">

<br>

<button class="btn btn-primary">Generate Dashboard</button>

</form>

</div>

{% if table %}

<div class="row mt-4">

<div class="col-md-3">
<div class="card p-3">
<h5>Total Demand</h5>
{{demand}}
</div>
</div>

<div class="col-md-3">
<div class="card p-3">
<h5>Reorder Quantity</h5>
{{reorder_qty}}
</div>
</div>

<div class="col-md-3">
<div class="card p-3">
<h5>Overstock</h5>
{{over_stock}}
</div>
</div>

<div class="col-md-3">
<div class="card p-3">
<h5>Accuracy</h5>
{{accuracy}} %
</div>
</div>

</div>

<div class="mt-4">

{{table | safe}}

</div>

<div class="mt-4">

<img src="data:image/png;base64,{{graph}}" class="img-fluid">

</div>

{% endif %}

</div>

</body>

</html>

"""

@app.route("/",methods=["GET","POST"])
def home():

    table=None

    if request.method=="POST":

        file=request.files["file"]
        stock=int(request.form["stock"])

        df=pd.read_csv(file)

        forecast,accuracy,predictions = forecast_sales(df)

        status,reorder_qty,over_stock,demand = inventory_analysis(stock,forecast)

        forecast_df=pd.DataFrame({
            "Day":range(1,31),
            "Forecast Sales":forecast
        })

        table=forecast_df.to_html(classes="table table-striped")

        plt.figure()

        plt.plot(df["sales"],label="Actual")
        plt.plot(predictions,label="Predicted")
        plt.plot(range(len(df),len(df)+30),forecast,label="Forecast")

        plt.legend()

        img=io.BytesIO()
        plt.savefig(img,format="png")
        img.seek(0)

        graph=base64.b64encode(img.getvalue()).decode()

        accuracy=round(accuracy*100,2)

        return render_template_string(html,
        table=table,
        reorder_qty=reorder_qty,
        over_stock=over_stock,
        demand=demand,
        accuracy=accuracy,
        graph=graph)

    return render_template_string(html)


port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)