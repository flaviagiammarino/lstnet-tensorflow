import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df, n_targets):

    '''
    Plot the actual and predicted values of the time series.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with actual and predicted values of the time series.

    n_targets: int.
        Number of time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of actual and predicted values of the time series,
        one subplot for each time series.
    '''

    fig = make_subplots(
        subplot_titles=['Target ' + str(i + 1) for i in range(n_targets)],
        vertical_spacing=0.15,
        rows=n_targets,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#000000',
            ),
        ),
    )

    fig.update_annotations(
        font=dict(
            size=13
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
                    color='#b3b3b3',
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
                    color='#0550ae',
                ),
            ),
            row=i + 1,
            col=1
        )

        fig.update_xaxes(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )

    return fig
