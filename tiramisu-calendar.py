from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import streamlit as st
from streamlit_calendar import calendar


def transform_awt_to_activity_log(dataframe, inactivity_threshold=pd.Timedelta("1m")):
    """Transforms an active window tracking log into an activity log

    To transform it, we group all active windows events until a change happens. We consider it a change if the activity or the case assigned
    to the active window is different or if the inactivity period between two active windows (the difference between the end of one event
    and the beginning of the next one) is above the inactivity_threshold. At this moment we only keep the first window title.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe with the active window tracking event log (as exported by Tockler together with info about activity and case)
    inactivity_threshold: Timedelta
        The threshold to consider a new activity

    Returns
    -------
    dataframe
        The dataframe with the activity log
    """
    # We consider it a change if the activity is different or if the gap between the end of an activity and the beginning of the next is greater than a threshold
    change = (((dataframe["WP flow activity"].shift() != dataframe["WP flow activity"]) & (dataframe["Case"].shift() != dataframe["Case"]))  |  ((dataframe["Begin"] - dataframe["End"].shift()) > inactivity_threshold))
    it = change.cumsum()

    pr = dataframe.groupby(by=it).agg({"Begin": "first", "End": "last", "WP flow activity": "first", "Case":"first", "Title": "first"})
    pr["Duration"] = pr["End"] - pr["Begin"]
    pr["Duration_minutes"] = pr["Duration"] / pd.Timedelta('1m')
    prev = pr["WP flow activity"].shift()
    prev.loc[pr["Begin"].dt.date != pr["Begin"].shift().dt.date] = np.nan
    pr["Prev"] = prev
    # We consider the gap only within the same day (alternatively, we could also consider there is a gap before the first activity in the morning)
    pr["Gap"] = ((pr["Begin"] - pr.shift()["End"] > inactivity_threshold) & (pr["Begin"].dt.day == pr.shift()["End"].dt.day))    

    return pr


def expand_events(dataframe, time_slot='1H'):
    """ Splits events in the event log based on the time intervals determined by the time slot

    For instance, if the time slot is '1H' (one hour), and there is an event that begins at
    8:58 and ends at 9:04, it splits the event in two: one from 8:58 to 9:00 and another one
    from 9:00 to 9:04. 

    Parameters
    ----------
    dataframe: DataFrame
        The dataframe with the event log
    time_slot: str, optional
        The time interval used to split the event log using the same values as in pandas Timedelta (default '1H')

    Returns
    -------
    dataframe
        The dataframe with the splitted events
    """
    expanded_rows = []
    
    for index, row in dataframe.iterrows():
        current_time = row['Begin']
        end_time = row['End']
        
        #while current_time.hour < end_time.hour:
        while (current_time + pd.Timedelta(time_slot)).floor(time_slot) < end_time:
            new_time = (current_time + pd.Timedelta(time_slot)).floor(time_slot)
            expanded_rows.append({
                'Begin': current_time,
                'End': min(new_time, end_time),
                'WP flow activity': row['WP flow activity'],
                'Case': row['Case'],
                'Duration': min(new_time, end_time) - current_time
            })
            
            current_time = new_time

        if end_time != current_time:
            expanded_rows.append({
                'Begin': current_time,
                'End': end_time,
                'WP flow activity': row['WP flow activity'],
                'Case': row['Case'],
                'Duration': end_time - current_time
            })
    
    return pd.DataFrame(expanded_rows)

def compute_hourly_schedule(df, freq='1H', empty_threshold=None, misc_threshold=None):
    """Computes a new log in which each row represents a predetermined time interval instead of an activity

    This function receives a dataframe that represents an event log in which each row
    represents an activity and returns a new dataframe in which each row is a predetermined
    time interval instead of an activity. The time interval used is specified in a parameter. 
    There are two thresholds that configure how the activity that is executed in each time
    interval is determined. The empty_threshold is used to determine whether an activity has
    been executed in that time interval at all. If the duration of the  activities performed 
    in the time interval is lower than the empty_threshold, then the time interval is classified
    as '**Empty'. If the time interval is not empty, then the activity performed in the time
    interval is the activity with the greatest duration in that interval if the duration is
    above the misc_threshold or if there is only one activity performed in that interval. 
    Otherwise, the time interval is classified as '**Misc'. 
    
    Parameters
    ----------
    df : DataFrame
        The input dataframe with the activity event log
    freq : str, optional
        The size of the time interval using the same values as in pandas Timedelta (default is '1H')
    empty_threshold : Timedelta, optional
        The threshold to determine whether an activity has been executed in a time interval (default is freq/4)
    misc_threshold: Timedelta, optional
        The threshold to determine whether the time interval is assigned to an activity (default is freq/3)
        
    Returns
    -------
    dataframe
        a dataframe with a time interval event log
    """
    if empty_threshold is None:
        empty_threshold = pd.Timedelta(freq) / 4
    if misc_threshold is None:
        misc_threshold = pd.Timedelta(freq) / 3

    ee = expand_events(df, time_slot=freq)
    ee['Hour'] = ee['Begin'].dt.floor(freq)
    act_per_hour = ee.groupby(['Hour', 'WP flow activity', 'Case'])["Duration"].sum().reset_index()
    max_per_hour = act_per_hour.groupby('Hour')["Duration"].max()
    num_act_per_hour = act_per_hour.groupby('Hour')["Duration"].count()
    sum_per_hour = act_per_hour.groupby('Hour')["Duration"].sum()
    idx = act_per_hour.groupby('Hour')["Duration"].transform('max') == act_per_hour["Duration"]
    act_max_per_hour = act_per_hour[idx].set_index('Hour')
    act_max_per_hour['Total'] = sum_per_hour

    act_max_per_hour.loc[(max_per_hour < misc_threshold) & (num_act_per_hour > 1), 'WP flow activity'] = '**Misc'
    act_max_per_hour.loc[(max_per_hour < misc_threshold) & (num_act_per_hour > 1), 'Case'] = '**Misc'
    act_max_per_hour.loc[sum_per_hour < empty_threshold, 'WP flow activity'] = '**Empty'
    act_max_per_hour.loc[sum_per_hour < empty_threshold, 'Case'] = '**Empty'

    return act_max_per_hour
    

def group_hours(dataframe, slot='1H'):
    """Groups all consecutive time intervals with the same activity

    Parameters
    ----------
    dataframe : DataFrame
        A dataframe that contains a time interval event log
    slot : str, optional
        The slot used in the time interval event log (default is '1H'). This is necessary
        because there is no safe way to compute this from the event log.

    Returns
    -------
    dataframe
        The dataframe with the time interval event log 
    """
    hs_log = dataframe.reset_index()
    hs_log['NextHour'] = hs_log['Hour'] + pd.Timedelta(slot)
    hs_change = (((hs_log["WP flow activity"].shift() != hs_log["WP flow activity"]) | (hs_log["Case"].shift() != hs_log["Case"])  |  ((hs_log['Hour'].dt.day.shift() != hs_log['Hour'].dt.day))))
    it = hs_change.cumsum()
    pr = hs_log.groupby(by=it).agg(Begin=("Hour", "first"), End= ("NextHour", "last"), Activity=("WP flow activity", "first"), Case=("Case", "first"))

    return  pr


def compute_metrics(dataframe, activity=None):

    if activity is not None:
        effective_duration = dataframe[dataframe["WP flow activity"]==activity]["Duration"].sum() / pd.Timedelta('1min')
        other_activities = (dataframe["Duration"].sum() / pd.Timedelta('1min')) - effective_duration    
        times_resumed = dataframe[dataframe["WP flow activity"]==activity]["Duration"].count()
        mean_slot_duration = dataframe[dataframe["WP flow activity"]==activity]["Duration"].mean() / pd.Timedelta('1min')
    else:
        effective_duration = dataframe["Duration"].sum() / pd.Timedelta('1min')
        other_activities = 0
        times_resumed = dataframe["Duration"].count()
        mean_slot_duration = dataframe["Duration"].mean() / pd.Timedelta('1min')        

    total_duration = (dataframe["End"].max() - dataframe["Begin"].min()) / pd.Timedelta('1min')
    number_activities = dataframe["WP flow activity"].nunique()
    percentage_effective = float(effective_duration) / float(total_duration)
    external_interruptions = total_duration - (effective_duration + other_activities)

    return {
        "effective_duration": effective_duration,
        "percentage_effective": percentage_effective,
        "other_activities": other_activities,
        "total_duration": total_duration,
        "external_interruptions": external_interruptions,
        "times_resumed": times_resumed,
        "mean_slot_duration": mean_slot_duration,
        "number_activities": number_activities
    }


@st.cache_data
def load_data(awt_data):
    ifull = pd.read_csv(awt_data, delimiter=";", parse_dates=True, infer_datetime_format=True)
    ifull["Begin"] = pd.to_datetime(ifull["Begin"], format="%d-%m-%Y %H:%M")
    ifull["End"] = pd.to_datetime(ifull["End"], format="%d-%m-%Y %H:%M")    
    pr = transform_awt_to_activity_log(ifull)

    return pr

@st.cache_data
def load_calendar(calendar_data):
    calendar_csv = pd.read_csv(calendar_data, delimiter=',')
    caldf = pd.DataFrame(calendar_csv[["Subject", "All day event"]])
    caldf["Start"] = pd.to_datetime(calendar_csv["Start Date"] + " " + calendar_csv["Start Time"], format="%d-%m-%Y %H:%M:%S")
    caldf["End"] = pd.to_datetime(calendar_csv["End Date"] + " " + calendar_csv["End Time"], format="%d-%m-%Y %H:%M:%S")
    

    return caldf

def cal_to_calendar(caldf):
    return [{
        "title": row['Subject'],
        "start": row['Start'].isoformat(),
        "end": row['End'].isoformat(),
        "backgroundColor": "#DDDDDD",
        "textColor": "#000000",
        "allDay": row["All day event"],
        "extendedProps": {
            "calendar_event": True
        }
    } for i,row in caldf.iterrows()]

    

def hourly_schedule(pr):
    return group_hours(compute_hourly_schedule(pr, freq='15min'), slot='15min')

def to_calendar_format(dataframe, palette, activities=[]):
    df = dataframe[dataframe['Activity'] != "**Empty"]
    return [{
                "title": row['Activity'] + " - " + row['Case'],
                "start": row['Begin'].isoformat(),
                "end": row['End'].isoformat(),
                "backgroundColor": palette[row['Activity']],
                "extendedProps": {
                    "activity": row['Activity'],
                    "case": row['Case']
                } 
            } for i,row in df.iterrows() if (row['Activity'] in activities if len(activities) >0 else True)]

def create_color_palette(pr):
    activities = np.append(pr["WP flow activity"].unique(),"**Misc")
    pal = sns.color_palette(palette="tab20b", n_colors=len(activities)).as_hex()
    return {a: p for a,p in zip(activities, pal)}


st.set_page_config(layout="wide")
st.title('Calendar of activities')

with st.sidebar:
    awt_data = st.file_uploader(label='Active window tracking data')
    calendar_data = st.file_uploader(label='Calendar data')

if awt_data is not None:  
    data_load_state = st.text('Loading data...')
    raw = load_data(awt_data)
    palette = create_color_palette(raw)
    hourly = hourly_schedule(raw)
    data_load_state.text("Done!")

if calendar_data is not None:
    caldf = load_calendar(calendar_data)


    # with st.sidebar:
    #     st.header("Explore activities")
    #     search = st.selectbox(label="Search activities:", options=palette.keys())
    #     calendar(
    #             events = to_calendar_format(hourly, palette, [search]), 
    #             options = {
    #                 # "slotMinTime": "08:00:00",
    #                 # "slotMaxTime": "18:00:00",
    #                 "headerToolbar": {
    #                     "right": "prev,next"
    #                 },
    #                 "initialView": "listYear",
    #                 "initialDate": "2023",
    #                 "firstDay": 1,
    #                 "height": "500px"
    #             }, key='calendar_sidebar')        

    # st.sidebar.dataframe(
    #     hourly[hourly['Activity']==search][["Begin", "End", "Case"]], 
    #     hide_index=True, 
    #     height=500,
    #     column_config={
    #         "Begin": st.column_config.DatetimeColumn("Begin", format="D MMM YYYY, hh:mm"),
    #         "End": st.column_config.DatetimeColumn("End", format="D MMM YYYY, hh:mm")
    #     })
def create_timeline(palette, cp):
    timeline_data = pd.DataFrame(cp[["Begin", "End"]])
    timeline_data['Title'] = cp["WP flow activity"] + " - " + cp["Case"]
    timeline_data['Color'] = cp["WP flow activity"].apply(lambda x: palette[x])
    timeline = alt.Chart(timeline_data).mark_bar().encode(
                    x=alt.X('Begin', axis=alt.Axis(title="", grid=True, format='%H:%M:%S')),
                    x2=alt.X2('End', title=""),
                    y=alt.Y('Title', axis=alt.Axis(title="")),
                    color=alt.Color('Color', scale=None, legend=None),
                    tooltip=[alt.Tooltip('Begin:T', format='%H:%M'), alt.Tooltip('End:T', format='%H:%M'), 'Title']
                )
    
    return timeline

def interval_details(raw, palette, event, start, end, time_format):
    time_filter = ((raw["Begin"] >= start) & (raw["Begin"] <= end)) | ((raw["End"] >= start) & (raw["End"] <= end))
    cp = raw[time_filter][["Begin", "End", "WP flow activity", "Case", "Duration"]].copy()
    cp.loc[cp["Begin"] < start, "Begin"] = start
    cp.loc[cp["End"] > end, "End"] = end
    main_activity = event["activity"] if event is not None and event["activity"] != "**Misc" else None

    st.subheader(f"Activities from {start.strftime(time_format)} to {end.strftime(time_format)}")
    if main_activity is not None:
        st.markdown(f"##### Main activity: {event['activity']} - {event['case']}")

    #st.write(f"Starts at {cp['Begin'].min()} and ends at {cp['End'].max()}")

    timeline = create_timeline(palette, cp)
    st.altair_chart(timeline, use_container_width=True)

    metrics = compute_metrics(cp, main_activity)

    if main_activity is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Effective duration", f'{round(metrics["effective_duration"],2):g} min') 
        col2.metric("Other activities", f'{round(metrics["other_activities"],2):g} min')
        col3.metric("External interruptions", f'{round(metrics["external_interruptions"],2):g} min')
        col4, col5, col6 = st.columns(3)
        col4.metric("Percentage effective", f'{round(metrics["percentage_effective"]*100,2):g} %')
        col5.metric("Times resumed", metrics["times_resumed"])
        col6.metric("Number other activities", metrics["number_activities"]-1)
    else:
        col1, col2= st.columns(2)
        col1.metric("Effective duration", f'{round(metrics["effective_duration"],2):g} min') 
        col2.metric("External interruptions", f'{round(metrics["external_interruptions"],2):g} min')
        col4, col5 = st.columns(2)
        col4.metric("Percentage effective", f'{round(metrics["percentage_effective"]*100,2):g} %')
        col5.metric("Number different activities", metrics["number_activities"])


    with st.expander("See details of the interval:"):
        st.dataframe(
                        cp,            
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Begin": st.column_config.DatetimeColumn("Begin", format="HH:mm"),
                            "End": st.column_config.DatetimeColumn("End", format="HH:mm")
                        })

if awt_data is None:
    st.write("You need to upload the data first")
else:
    filter_selection = st.multiselect(label='Filter by activity:', options=palette.keys(), key='selection')

    if calendar_data is not None:
        include_calendar = st.checkbox(label='Include calendar data', value=True)
    else:
        include_calendar = False

    col_calendar, col_details = st.columns([0.5, 0.5], gap="medium")

    with col_calendar:
        data = to_calendar_format(hourly, palette, filter_selection)

        calendar_options = {
            "editable": False,
            "selectable": True,
            "headerToolbar": {
                "left": "today prev,next",
                "center": "title",
                "right": "timeGridWeek, timeGridDay, dayGridMonth, listYear",
            },
            # "slotMinTime": "08:00:00",
            # "slotMaxTime": "18:00:00",
            "slotDuration": "00:15:00",
            "initialView": "timeGridWeek",
            "initialDate": "2023-03-06",
            "height": 600,
            "firstDay": 1
        }

        custom_css="""
            .fc-event-past {
                opacity: 0.8;
            }
            .fc-event-time {
                font-size: 0.75em;
                font-style: italic;
                display: none;
            }
            .fc-event-title {
                font-weight: 70;
                font-size: 0.75em;
            }
            .fc-toolbar-title {
                font-size: 1em;
            }
        """

        if include_calendar:
            data = data + cal_to_calendar(caldf)
            calendar_sel = calendar(events = data, options = calendar_options, custom_css = custom_css, key=f"cal_with_cal_{hash(tuple(filter_selection))}")
        else:
            calendar_sel = calendar(events = data, options = calendar_options, custom_css = custom_css, key=f"cal_without_cal_{hash(tuple(filter_selection))}")        

    if "callback" in calendar_sel and calendar_sel["callback"] == "eventClick":
        event = calendar_sel["eventClick"]["event"]
        start = datetime.fromisoformat(event['start']).replace(tzinfo=None)
        end = datetime.fromisoformat(event['end']).replace(tzinfo=None)
        time_format = "%Y-%m-%d %H:%M"

        with col_details:            
            #st.write(details_event)
            if "calendar_event" in event["extendedProps"]:
                st.subheader(f"Calendar event: {event['title']}")
                st.write(f"From {start.strftime(time_format)} to {end.strftime(time_format)}")
            else:
                interval_details(raw, palette, event["extendedProps"], start, end, time_format)
            
        with col_calendar:
            if "activity" in event["extendedProps"]:
                st.write(f'Other instances of activity {event["extendedProps"]["activity"]}:')
                    # st.write()
                    # st.dataframe(
                    #     hourly[hourly['Activity']==event["extendedProps"]["activity"]][["Begin", "End", "Case"]], 
                    #     hide_index=True, 
                    #     height=500,
                    #     column_config={
                    #         "Begin": st.column_config.DatetimeColumn("Begin", format="D MMM YYYY, hh:mm"),
                    #         "End": st.column_config.DatetimeColumn("End", format="D MMM YYYY, hh:mm")
                    #     })
                    
                data_detail = to_calendar_format(hourly, palette, [event["extendedProps"]["activity"]])
                details_event = calendar(
                    events = data_detail, 
                    options = {
                        "headerToolbar": {
                            "right": "prev,next"
                        },
                        "initialView": "listYear",
                        "initialDate": "2023",
                        "firstDay": 1
                    },                 
                    custom_css = custom_css,
                    key = f'calendar_detail_{event["extendedProps"]["activity"]}')      

    elif "callback" in calendar_sel and calendar_sel["callback"] == "select":
        start = datetime.strptime(calendar_sel['select']['start'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=None)
        end = datetime.strptime(calendar_sel['select']['end'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=None)
        time_format = "%Y-%m-%d %H:%M"

        with col_details:            
            interval_details(raw, palette, None, start, end, time_format)


