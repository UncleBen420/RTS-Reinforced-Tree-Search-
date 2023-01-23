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
        nb_actions = [0]
        app = dash.Dash(__name__)
        app.title = 'RTS learning realtime monitor'

        episodes = range(len(vs))
        fig_g = px.line(x=episodes, y=vs, title='overall good action taken over bad.',
                        labels=dict(x="Episodes", y="Good/Bad ratio"))
        fig_r = px.line(x=episodes, y=rewards, title='Rewards',
                        labels=dict(x="Episodes", y="Rewards"))
        fig_l = px.line(x=episodes, y=losses, title='Loss',
                        labels=dict(x="Episodes", y="Loss"))
        fig_nb = px.line(x=episodes, y=nb_actions, title='Number of action took',
                         labels=dict(x="Episodes", y="NB actions"))

        app.layout = html.Div(
            [
                dcc.Graph(id='fig_g', figure=fig_g),
                dcc.Graph(id='fig_r', figure=fig_r),
                dcc.Graph(id='fig_l', figure=fig_l),
                dcc.Graph(id='fig_nb', figure=fig_nb),
                dcc.Interval(id='graph-update', interval=10000, n_intervals=0)
            ]
        )

        @app.callback([
            Output('fig_g', 'figure'),
            Output('fig_r', 'figure'),
            Output('fig_l', 'figure'),
            Output('fig_nb', 'figure')],
            [Input('graph-update', 'n_intervals')]
        )
        def update_graph_scatter(n):
            while True:
                try:
                    v, r, l, td, nb = q.get(block=False)
                    vs.append(v)
                    rewards.append(r)
                    losses.append(l)
                    nb_actions.append(nb)
                except queue.Empty:
                    break

            episodes = range(len(vs))
            fig_g = px.line(x=episodes, y=vs, title='Mean state value (V)',
                            labels=dict(x="Episodes", y="Mean of V"))
            fig_r = px.line(x=episodes, y=rewards, title='Rewards',
                            labels=dict(x="Episodes", y="Rewards"))
            fig_l = px.line(x=episodes, y=losses, title='Loss',
                            labels=dict(x="Episodes", y="Loss"))
            fig_nb = px.line(x=episodes, y=nb_actions, title='Number of action took',
                             labels=dict(x="Episodes", y="NB actions"))

            return fig_g, fig_r, fig_l, fig_nb

        app.run()

    def update_values(self, v, r, l, nb):
        try:
            self.q.put((v, r, l, nb), block=False)
        except queue.Full:
            pass

    def stop_server(self):
        self.server.terminate()
        self.server.join()

    def start_server(self):
        self.q = Queue()
        self.server = Process(target=self.work, args=(self.q,))
        self.server.start()


def metrics_to_pdf(g, r, l, nb, nbc, mza, directory, stage):

    good = np.array(g)
    bad = 1 - good
    base = np.linspace(0, 1, len(good))

    pp = PdfPages(directory + stage + "_Good.pdf")
    plt.clf()
    plt.plot(base, good + bad, label="good hits", color='g')
    plt.plot(base, bad, label="bad hits", color='r')
    plt.fill_between(base, bad, good + bad, color='g', alpha=.5)
    plt.fill_between(base, 0, bad, color='r', alpha=.5)
    plt.title('Choice of good action over bad action')
    plt.xlabel('Episodes')
    plt.ylabel('Good/bad ratio')
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

    pp = PdfPages(directory + stage + "_nb_steps.pdf")
    plt.clf()
    plt.plot(nb, label="agent")
    plt.plot(nbc, label="conventional policy")
    plt.plot(mza, label="agent at the maximum zoom")
    plt.title('Number of steps during each episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend(loc='upper right')
    pp.savefig()
    pp.close()


def metrics_eval_to_pdf(g, r, nb, nbc, pertinence, precision, directory, stage):

    pp = PdfPages(directory + stage + "_Good.pdf")
    plt.clf()
    plt.boxplot(g)
    plt.title('Choice of good action over bad action')
    plt.ylabel('Good/Bad ratio')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_R.pdf")
    plt.clf()
    plt.boxplot(r)
    plt.title('Sum of rewards accumulated during an episode')
    plt.ylabel('Rewards')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_nb_steps.pdf")
    plt.clf()
    labels = ["agent", "conventional policy"]
    plt.boxplot([nb, nbc], labels=labels)
    plt.title('Number of steps during each episodes')
    plt.ylabel('Steps')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_precision.pdf")
    plt.clf()
    plt.boxplot(precision)
    plt.title('precision of the guess on the object position')
    plt.ylabel('precision')
    pp.savefig()
    pp.close()

    pp = PdfPages(directory + stage + "_pertinence.pdf")
    plt.clf()
    plt.boxplot(pertinence)
    plt.title('pertinence of using rts over a conventional policy')
    plt.ylabel('pertinence')
    pp.savefig()
    pp.close()
