import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df):

    '''
    Plot the actual and predicted values of the time series.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with actual and predicted values of the time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of actual and predicted values of the time series,
        one subplot for each time series.
    '''

    # get the number of targets
    n_targets = (df.shape[1] - 1) // 2

    # plot the forecasts for each target
    fig = make_subplots(
        subplot_titles=['Target ' + str(i + 1) for i in range(n_targets)],
        vertical_spacing=0.15,
        rows=n_targets,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )

    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )

    for i in range(n_targets):

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['actual_' + str(i + 1)],
                name='Actual',
                legendgroup='Actual',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    color='#afb8c1',
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['predicted_' + str(i + 1)],
                name='Forecast',
                legendgroup='Forecast',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    width=1,
                    color='#0969da',
                ),
            ),
            row=i + 1,
            col=1
        )

        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )

    return fig
