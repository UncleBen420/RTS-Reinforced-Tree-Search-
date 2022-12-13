import threading
import time
from flask import request

import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import plotly.express as px
import logging
from multiprocessing import Process, Queue
import queue
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

# Disable unnecessary log
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class MetricMonitor:

    def __init__(self):
        self.server = None
        self.q = None

    def work(self, q):

        vs = [0]
        rewards = [0]
        losses = [0]
        td_errors = [0]
        nb_actions = [0]
        app = dash.Dash(__name__)
        app.title = 'RTS learning realtime monitor'

        episodes = range(len(vs))
        fig_v = px.line(x=episodes, y=vs, title='Mean state value (V)',
                        labels=dict(x="Episodes", y="Mean of V"))
        fig_r = px.line(x=episodes, y=rewards, title='Rewards',
                        labels=dict(x="Episodes", y="Rewards"))
        fig_l = px.line(x=episodes, y=losses, title='Loss',
                        labels=dict(x="Episodes", y="Loss"))
        fig_td = px.line(x=episodes, y=td_errors, title='TD error',
                         labels=dict(x="Episodes", y="TDE"))
        fig_nb = px.line(x=episodes, y=nb_actions, title='Number of action took',
                         labels=dict(x="Episodes", y="NB actions"))

        app.layout = html.Div(
            [
                dcc.Graph(id='fig_v', figure=fig_v),
                dcc.Graph(id='fig_r', figure=fig_r),
                dcc.Graph(id='fig_l', figure=fig_l),
                dcc.Graph(id='fig_td', figure=fig_td),
                dcc.Graph(id='fig_nb', figure=fig_nb),
                dcc.Interval(id='graph-update', interval=1000, n_intervals=0)
            ]
        )

        @app.callback([
            Output('fig_v', 'figure'),
            Output('fig_r', 'figure'),
            Output('fig_l', 'figure'),
            Output('fig_td', 'figure'),
            Output('fig_nb', 'figure')],
            [Input('graph-update', 'n_intervals')]
        )
        def update_graph_scatter(n):

            try:
                v, r, l, td, nb = q.get(block=False)
                vs.append(v)
                rewards.append(r)
                losses.append(l)
                td_errors.append(td)
                nb_actions.append(nb)
            except queue.Empty:
                pass

            episodes = range(len(vs))
            fig_v = px.line(x=episodes, y=vs, title='Mean state value (V)',
                            labels=dict(x="Episodes", y="Mean of V"))
            fig_r = px.line(x=episodes, y=rewards, title='Rewards',
                            labels=dict(x="Episodes", y="Rewards"))
            fig_l = px.line(x=episodes, y=losses, title='Loss',
                            labels=dict(x="Episodes", y="Loss"))
            fig_td = px.line(x=episodes, y=td_errors, title='TD error',
                             labels=dict(x="Episodes", y="TDE"))
            fig_nb = px.line(x=episodes, y=nb_actions, title='Number of action took',
                             labels=dict(x="Episodes", y="NB actions"))

            return fig_v, fig_r, fig_l, fig_td, fig_nb

        app.run()

    def update_values(self, v, r, l, td, nb):
        try:
            self.q.put((v, r, l, td, nb), block=False)
        except queue.Full:
            pass

    def stop_server(self):
        self.server.terminate()
        self.server.join()

    def start_server(self):
        self.q = Queue()
        self.server = Process(target=self.work, args=(self.q,))
        self.server.start()


def metrics_to_pdf(v, r, l, td, nb, nbc, directory, stage):

    pp = PdfPages(directory + stage + "_V.pdf")
    plt.clf()
    plt.plot(v)
    plt.title('Mean of state value during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('V')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_R.pdf")
    plt.clf()
    plt.plot(r)
    plt.title('Sum of rewards accumulated during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_loss.pdf")
    plt.clf()
    plt.plot(l)
    plt.title('Loss during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_TDE.pdf")
    plt.clf()
    plt.plot(td)
    plt.title('Mean of TD error during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('TDE')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_nb_steps.pdf")
    plt.clf()
    plt.plot(nb, label="agent")
    plt.plot(nbc, label="conventional policy")
    plt.title('Number of steps during each episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend(loc='upper right')
    pp.savefig()
    pp.close()


def metrics_eval_to_pdf(v, r, nb, nbc, directory, stage):

    pp = PdfPages(directory + stage + "_V.pdf")
    plt.clf()
    plt.plot(v)
    plt.title('Mean of state value during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('V')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_R.pdf")
    plt.clf()
    plt.plot(r)
    plt.title('Sum of rewards accumulated during an episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_nb_steps.pdf")
    plt.clf()
    plt.plot(nb, label="agent")
    plt.plot(nbc, label="conventional policy")
    plt.title('Number of steps during each episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend(loc='upper right')
    pp.savefig()
    pp.close()
