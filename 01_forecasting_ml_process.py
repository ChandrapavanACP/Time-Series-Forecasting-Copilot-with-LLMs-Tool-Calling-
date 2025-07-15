# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Demonstrate the steps in making a forecast with machine learning

# LIBRARIES

import pandas as pd

import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"


# * STEP 1: Get some data

df_sample = pd.read_csv("data/walmart_sales_monthly.csv")

df_sample

# * STEP 2: Create the forecasting function

def forecast_ts(data, id_col = 'item_id', date_col = 'year_month', value_col = 'total_value', forecast_horizon = 12):
    
    # Wrap imports inside function to make scope local:
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from pytimetk import future_frame

    # Convert input dictionary to DataFrame
    df = pd.DataFrame(data)

    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Get unique item IDs
    unique_items = df[id_col].unique()

    # Wrap functions to enforce local scope for exec() command:
    def extend_single_timeseries_frame(data, id_col, date_col, value_col, id_value, h=60):        

        # Filter for one time series
        df_single = data[data[id_col] == id_value].copy()
        df_single[date_col] = pd.to_datetime(df_single[date_col])

        # Actual data with ACTUAL label
        df_actual = df_single.assign(key="ACTUAL")

        # Generate future dates using pytimetk
        df_future = (
            future_frame(df_single, date_column=date_col, length_out=forecast_horizon, bind_data=True)
            .assign(**{value_col: np.nan, "key": "FUTURE"})
            .tail(forecast_horizon)
        )

        # Combine actual and future
        combined_df = pd.concat([df_actual, df_future], ignore_index=True)

        return combined_df
    
    def make_timeseries_features(data, date_col):
        """Adds engineered time series features from the datetime column."""
        data['_index.num'] = pd.to_datetime(data[date_col]).astype(np.int64) // 10**9
        data['_year'] = pd.to_datetime(data[date_col]).dt.year
        data['_month'] = pd.to_datetime(data[date_col]).dt.month
        data['_mday'] = pd.to_datetime(data[date_col]).dt.day
        data['_wday'] = pd.to_datetime(data[date_col]).dt.dayofweek
        return data


    def forecast_single_timeseries(data, id_col, date_col, value_col, id_value, h=60, **kwargs):
        "Forecasts a single timeseries using XGBoost."
        single_timeseries_extended_df = extend_single_timeseries_frame(data, id_col, date_col, value_col, id_value, h)
        single_timeseries_feat_df = make_timeseries_features(single_timeseries_extended_df, date_col=date_col)

        future_df = single_timeseries_feat_df.query("key == 'FUTURE'")
        actual_df = single_timeseries_feat_df.query("key == 'ACTUAL'")

        df = actual_df.drop([id_col, 'key'], axis=1)
        X = df.drop([value_col, date_col], axis=1)
        y = df[value_col]

        # Calculate Conformal Interval
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=h, random_state=123, shuffle=False)

        model = XGBRegressor(**kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        std_dev = np.std(abs(y_test - y_pred))

        # Refit to full dataset
        X_train = actual_df.drop([id_col, 'key', value_col, date_col], axis=1)
        y_train = actual_df[value_col]

        model.fit(X_train, y_train)

        # Predict on future data
        future_df[value_col] = model.predict(future_df.drop([id_col, 'key', value_col, date_col], axis=1))

        future_df['conf_lo'] = future_df[value_col] - 1.96 * std_dev
        future_df['conf_hi'] = future_df[value_col] + 1.96 * std_dev

        combined_df = pd.concat([actual_df, future_df], axis=0)
        return combined_df

    # Initialize list for forecasted DataFrames
    forecast_dfs = []

    # Loop over each unique item and make forecasts
    for item in unique_items:
        forecast_df = forecast_single_timeseries(
            data=df,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            id_value=item,
            h=forecast_horizon
        )
        forecast_dfs.append(forecast_df)

    # Combine all forecasts into a single DataFrame
    combined_forecast_df = pd.concat(forecast_dfs, ignore_index=True)

    # Remove the engineered features
    combined_forecast_df = combined_forecast_df.drop(columns=[col for col in combined_forecast_df.columns if col.startswith('_')])

    # Reset the Index
    # combined_forecast_df = combined_forecast_df.reset_index(names=[date_col])

    return combined_forecast_df

# Call the forecast_ts() function on the sample dataset
forecasted_data = forecast_ts(
    data=df_sample, 
    id_col='item_id', 
    date_col='year_month', 
    value_col='total_value', 
    forecast_horizon=12
)

# Display the forecasted data
forecasted_data

forecasted_data.to_dict()

# forecasted_data.to_csv("data/walmart_sales_monthly_forecast.csv", index=False)

# * STEP 3: Plot the forecasted data

def plot_forecast(data, id_col='item_id', date_col='year_month', value_col='total_value'):
    import pandas as pd
    import plotly.graph_objs as go

    # Ensure the DataFrame is created from the provided data
    df = pd.DataFrame(data)

    # Define the required column names

    conf_lo_col = 'conf_lo'
    conf_hi_col = 'conf_hi'
    
    unique_ids = df[id_col].unique()

    # Create an initial figure
    fig = go.Figure()

    # Add traces for each item_id
    for i, item_id in enumerate(unique_ids):
        group = df[df[id_col] == item_id]

        # Add the main forecast line for each item
        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[value_col], 
            mode='lines', 
            showlegend=False,
            name='Forecast - ' + str(item_id),
            visible=True if i == 0 else False
        ))

        # Add the upper and lower bounds as filled area for each item
        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[conf_hi_col], 
            mode='lines', 
            name='Upper Bound - ' + str(item_id),
            line=dict(width=0),
            showlegend=False,
            visible=True if i == 0 else False
        ))

        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[conf_lo_col], 
            mode='lines', 
            name='Lower Bound - ' + str(item_id),
            fill='tonexty', 
            fillcolor='rgba(173,216,230,0.3)',  # Light blue fill
            line=dict(width=0),
            showlegend=False,
            visible=True if i == 0 else False
        ))

    # Create a dropdown to toggle between the forecasts
    dropdown_buttons = [
        dict(
            label=str(item_id),
            method="update",
            args=[dict(visible=[i // 3 == idx for i in range(len(unique_ids) * 3)]),
                   dict(title="Forecasts with Confidence Intervals - " + str(item_id))]
        )
        for idx, item_id in enumerate(unique_ids)
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                x=0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                showactive=True,
            )
        ],
        hovermode="x unified",
        title="Forecast - " + str(unique_ids[0]),
        xaxis_title=date_col,
        yaxis_title=value_col,
    )

    return fig

fig = plot_forecast(
    forecasted_data, 
    id_col='item_id', 
    date_col='year_month', 
    value_col='total_value'
)

fig

fig.to_dict()
