"""
author: Junming Huang
date: 7 Jul 2022
"""
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from pathlib import Path

def trend():
    # visualize
    trend = pandas.read_csv('data/covid1-trend.csv', index_col=0)
    trend.index = pandas.to_datetime(trend.index).to_series().dt.date
    fig, sentiment_ax = plt.subplots(figsize=(10, 8))
    volume_ax = sentiment_ax.twinx()  # instantiate a second axes that shares the same x-axis

    # front ax: overall sentiment
    rolling_window = 7
    sentiment_color = '#636363'
    sentiment_trend = trend[f'sentiment'].rolling(window=rolling_window, center=True).mean()
    sentiment_ax.plot(trend.index, trend[f'sentiment'].rolling(window=rolling_window, center=True).mean(), color=sentiment_color, zorder=30)
    sentiment_ax.fill_between(
            trend.index,
            sentiment_trend - trend['sentiment_ste'],
            sentiment_trend + trend['sentiment_ste'],
            color=sentiment_color, alpha=0.1, zorder=20)
    sentiment_ax.set_ylabel(f'Favorability ({rolling_window}-day rolling average)')
    sentiment_ax.yaxis.label.set_color(sentiment_color)
    sentiment_ax.tick_params(axis='y', colors=sentiment_color)
    sentiment_ax.spines['left'].set_edgecolor(sentiment_color)
    sentiment_ax.set(
        xlim=(datetime.date(2019,11,1), datetime.date(2020,6,30)),
        ylim=(None, -15))
    # sentiment_ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    # back ax: overall volume
    rolling_window = 3
    volume_ax.stackplot(trend.index,
                        trend['co_volume'].rolling(window=rolling_window, center=True).mean(),
                        trend['cv_volume'].rolling(window=rolling_window, center=True).mean(),
                        labels=['Daily volume of users mentioning China but not COVID-19',
                                'Daily volume of users mentioning China and COVID-19'],
                        colors=['#bdd7e7', '#3182bd'],
                        alpha=0.5)
    volume_ax.yaxis.label.set_color('#3182bd')
    volume_ax.tick_params(axis='y', colors='#3182bd')
    volume_ax.spines['right'].set_edgecolor('#3182bd')
    volume_ax.set_ylabel(f'Daily volume of users mentioning China ({rolling_window}-day rolling average)')
    volume_ax.spines['left'].set_visible(False)
    volume_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    volume_ax.set_ylim(top=80000)

    sentiment_ax.set_zorder(volume_ax.get_zorder() + 1)
    sentiment_ax.patch.set_visible(False)
    legend_lines = [
        matplotlib.lines.Line2D([0], [0], color=sentiment_color, label='Favorability'),
        matplotlib.patches.Patch(
            facecolor='#3182bd',
            edgecolor='white',
            label='Daily volume of users mentioning China and COVID-19'),
        matplotlib.patches.Patch(
            facecolor='#bdd7e7',
            edgecolor='white',
            label='Daily volume of users mentioning China but not COVID-19'),
    ]
    sentiment_ax.legend(handles=legend_lines, frameon=False, loc='upper right')

    fig.tight_layout()
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
    fig.savefig('figures/covid1-trend.pdf', transparent=False, dpi=100)
    fig.savefig('figures/covid1-trend.png', transparent=False, dpi=100)
    print('figures/covid1-trend.pdf')

def rd():
    # visualize
    effect_rd = pandas.read_csv('data/covid1-effect-rd.csv', header=0)
    effect_rd['treatment_week'] = pandas.to_datetime(effect_rd['treatment_week']).dt.date
    effect_rd = effect_rd.set_index(['treatment_week', 'size'])
    effect_rd.columns = pandas.to_datetime(effect_rd.columns).to_series().dt.date    
    color_map = plt.get_cmap('plasma')
    fig, ax = plt.subplots(figsize=(10, 8))
    xlim = (-4, 6)  # report prior 4 weeks to post 7 weeks

    treatment_weeks = effect_rd.index.get_level_values('treatment_week')[1:]
    treated_aligned = pandas.DataFrame(index=treatment_weeks[1:], columns=range(-20, 20), dtype=float)
    for k, (previous_week, treatment_week) in enumerate(zip(treatment_weeks[:-1], treatment_weeks[1:])):
        r = effect_rd.xs(key=treatment_week, level='treatment_week')
        treated = pandas.Series(index=((r.columns - treatment_week).days / 7).astype(int), data=r.values.squeeze())
        color = color_map(k / len(treatment_weeks))
        treated_aligned.loc[treatment_week, :] = treated
        ax.plot(treated[treated.index>=0].index, treated[treated.index>=0].values, color=color, alpha=0.1)
        ax.plot(treated[treated.index<=0].index, treated[treated.index<=0], '--', color=color, alpha=0.1)

    # merged: weighted average across weeks
    treated_aligned_series = pandas.Series(
        index=range(xlim[0], xlim[1]+1),
        data=numpy.average(treated_aligned.loc[:, xlim[0]:xlim[1]].values, axis=0, weights=effect_rd.reset_index().set_index('treatment_week').loc[treated_aligned.index, 'size']),
    )
    ax.plot(treated_aligned_series.loc[-1:].index, treated_aligned_series.loc[-1:], color='#3182bd', alpha=0.7, lw=3, label='Favorability after the treatment', zorder=10)
    ax.plot(treated_aligned_series.loc[:0].index, treated_aligned_series.loc[:0], ':', color='#3182bd', alpha=0.7, lw=3, label='Favorability before the treatment', zorder=20)

    # polish
    ax.set_ylim(-41, -5)
    ax.set_xlim(xlim)
    ax.set_xticks([-4, -1, 0, 1, 2, 3, 6])
    ax.set_xticklabels([
        '4 weeks\nbefore',
        '1 week\nbefore',
        'Treatment',
        '1 week\nafter',
        '2 weeks\nafter',
        '3 weeks\nafter',
        '6 weeks\nafter'])
    ax.axvline(x=-1, color='k', alpha=0.5)
    ax.axvline(x=0, color='k', alpha=0.5)
    ax.text(x=0 - 0.05, y=ax.get_ylim()[0] + 1.0, s='Treatment week',
            rotation=90, verticalalignment='bottom', horizontalalignment='right', alpha=0.5)

    # highlight the difference
    ax.annotate(text='',
                    xy=(0.3, treated_aligned_series[-1]),
                    xytext=(0.3, treated_aligned_series[0]),  # actually there is no text. this is other end.
                    xycoords='data', textcoords='data',
                    arrowprops={'arrowstyle': '|-|', 'color': '#3182bd'})
    ax.text(x=0.1 + 0.5,
                y=treated_aligned_series[1] + 2.0,
                s=f'Decline={treated_aligned_series[0] - treated_aligned_series[-1]:.2f}',
                rotation=90, color='#3182bd', ha='left', va='bottom')

    ax.set_ylabel(f"Favorability")
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/covid1-effect-rd.pdf', transparent=False, dpi=100)
    fig.savefig('figures/covid1-effect-rd.png', transparent=False, dpi=100)
    print('figures/covid1-effect-rd.pdf')

def did():
    effect_did = pandas.read_csv('data/covid1-effect-did.csv', header=0)
    effect_did['treatment_week'] = pandas.to_datetime(effect_did['treatment_week']).dt.date
    effect_did = effect_did.set_index(['treatment_week', 'group', 'size'])
    effect_did.columns = pandas.to_datetime(effect_did.columns).to_series().dt.date

    # visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    xlim = (-4, 6)  # report prior 4 weeks to post 7 weeks
    color_map = plt.get_cmap('plasma')

    treatment_weeks = effect_did.index.get_level_values('treatment_week').unique().sort_values()[1:]
    treated_aligned = pandas.DataFrame(index=treatment_weeks[1:], columns=range(-20, 20), dtype=float)
    control_aligned = pandas.DataFrame(index=treatment_weeks[1:], columns=range(-20, 20), dtype=float)

    for k, (previous_week, treatment_week) in enumerate(zip(treatment_weeks[:-1], treatment_weeks[1:])):
        color = color_map(k / len(treatment_weeks))
        """ r =
            2019-12-01   -28.270129
            2019-12-08   -26.002856
            2019-12-15   -28.352476
            2019-12-22   -23.675102
            2019-12-29   -25.428150
            2020-01-05   -22.380334
            2020-01-12   -17.094443
            2020-01-19   -21.960210
            2020-01-26   -27.716565
            2020-02-02   -26.589073
            2020-02-09   -34.064528
            2020-02-16   -29.998172
            2020-02-23   -28.508352
            2020-03-01   -26.774207
            2020-03-08   -28.819686
            2020-03-15   -37.286841
            2020-03-22   -35.131307
            2020-03-29   -36.784464
            2020-04-05   -36.992130
            2020-04-12   -37.652218
            2020-04-19   -35.396917
            2020-04-26   -36.164777
            2020-05-03   -35.315118
            Name: 49643, dtype: float64
        """
        r = effect_did.xs(key=treatment_week, level='treatment_week').xs(key='treated', level='group').squeeze()
        r.index = ((r.index - treatment_week).days / 7).astype(int)
        treated_aligned.loc[treatment_week, :] = r
        ax.plot(r.index, r.values, color=color, alpha=0.1)
        
        r = effect_did.xs(key=treatment_week, level='treatment_week').xs(key='control', level='group').squeeze()
        r.index = ((r.index - treatment_week).days / 7).astype(int)
        r = r[r.index <= 0]
        control_aligned.loc[treatment_week, :] = r
        ax.plot(r.index, r.values, '--', color=color, alpha=0.1)

    # merged: weighted average across weeks
    treated_aligned_series = pandas.Series(
        index=range(xlim[0], xlim[1]+1),
        data=numpy.average(treated_aligned.loc[:, xlim[0]:xlim[1]].values, axis=0, weights=effect_did.xs(key='treated', level='group').reset_index().set_index('treatment_week').loc[treated_aligned.index, 'size']),
    )
    control_aligned_series = pandas.Series(
        index=range(xlim[0], 0+1),
        data=numpy.average(control_aligned.loc[:, xlim[0]:0].values, axis=0, weights=effect_did.xs(key='control', level='group').reset_index().set_index('treatment_week').loc[control_aligned.index, 'size']),
    )
    ax.plot(treated_aligned_series.index, treated_aligned_series.values, color='#3182bd', alpha=0.7, lw=3, label='Favorability after the treatment', zorder=10)
    ax.plot(control_aligned_series.index, control_aligned_series.values, ':', color='#3182bd', alpha=0.7, lw=3, label='Favorability before the treatment', zorder=20)

    difference_in_difference = (treated_aligned_series[0] - treated_aligned_series[-1]) - (control_aligned_series[0] - control_aligned_series[-1])
    ax.fill_between(x=[-1, 0],
                        y1=[treated_aligned_series[-1],
                            treated_aligned_series[0]],
                        y2=[treated_aligned_series[-1],
                            treated_aligned_series[-1] - control_aligned_series[-1] + control_aligned_series[0]],
                        color='#3182bd', alpha=0.3,
                        label=f'Difference in difference = {difference_in_difference:.2f}')

    # polish
    ax.set_ylim(-41, -5)
    ax.set_xlim(xlim)
    ax.set_xticks([-4, 1, 0, 1, 2, 3, 6])
    ax.set_xticklabels([
        '4 weeks before\ntreatment',
        '1 week before\ntreatment',
        'Treatment',
        '1 week after\ntreatment',
        '2 weeks after\ntreatment',
        '3 weeks after\ntreatment',
        '6 weeks after\ntreatment'])
    ax.axvline(x=0, color='k', alpha=0.5)
    ax.axvline(x=1, color='k', alpha=0.5)
    ax.text(x=1 - .05, y=ax.get_ylim()[0] + 1.0, s='Treatment week',
            rotation=90, horizontalalignment='right', verticalalignment='bottom', alpha=0.5)
    # for ax in [ax_natural, ax_main, ax_details]:
    ax.set_ylabel(f"Favorability")
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('figures/covid1-effect-did.pdf', transparent=False, dpi=100)
    fig.savefig('figures/covid1-effect-did.png', transparent=False, dpi=100)
    print('figures/covid1-effect-did.pdf')

if __name__ == '__main__':
    trend()
    rd()
    did()