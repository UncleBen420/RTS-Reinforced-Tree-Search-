import threading
import time
from flask import request

import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import random
import plotly.express as px
import logging
from multiprocessing import Process, Queue
import queue



# Disable unnecessary log
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def work(q):

    labels = {"Mean state value (V)": [0],
              "Rewards": [0],
              "Loss": [0],
              "TD error": [0],
              "Number of action took": [0]}


    X = range(len(labels["Rewards"]))

    app = dash.Dash(__name__)

    fig = px.line(x=X, y=labels["Rewards"], title='A simple line graph')

    app.layout = html.Div(
        [
            html.Div(dcc.Dropdown(id='dropdown', options=list(labels.keys()))),
            dcc.Graph(id='fig1', figure=fig),
            dcc.Interval(id='graph-update', interval=1000, n_intervals=0)

        ]
    )

    @app.callback(
        Output('fig1', 'figure'),
        [Input('graph-update', 'n_intervals'), Input("dropdown", "value")]
    )
    def update_graph_scatter(n, label):

        try:
            v, r, l, td, nb = q.get(block=False)
            labels["Mean state value (V)"].append(v)
            labels["Rewards"].append(r)
            labels["Loss"].append(l)
            labels["TD error"].append(td)
            labels["Number of action took"].append(nb)
        except queue.Empty:
            pass

        if label is None:
            label = "Rewards"

        return px.line(x=range(len(labels[label])),
                       y=labels[label],
                       title=label)

    app.run()

def update_values(v, r, l, td, nb):
    try:
        print("coco")
        q.put((1,2,3,5,6), block=False)
    except queue.Full:
        pass

def stop_server():
    server.terminate()
    server.join()

def start_server():
    q = Queue()
    server = Process(target=work, args=(q,))
    server.start()




if __name__ == '__main__':


    time.sleep(2)

        pass
    time.sleep(20)
    print("yeah")


