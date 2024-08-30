from datetime import datetime

import altair as alt
import calplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_calendar import calendar

import awt


def load_data(awt_data):
    ifull = pd.read_csv(awt_data, delimiter=";", parse_dates=True, infer_datetime_format=True)
    ifull["Begin"] = pd.to_datetime(ifull["Begin"], format="%d-%m-%Y %H:%M")
    ifull["End"] = pd.to_datetime(ifull["End"], format="%d-%m-%Y %H:%M")    
    pr = awt.transform_awt_to_activity_log(ifull)

    return pr

def load_sample():
    ifull = pd.read_csv("https://raw.githubusercontent.com/project-pivot/labelled-awt-data/main/data/awt_data_1_pseudonymized.csv", delimiter=";", parse_dates=True, infer_datetime_format=True)
    ifull["Begin"] = pd.to_datetime(ifull["Begin"], format="%d-%m-%Y %H:%M")
    ifull["End"] = pd.to_datetime(ifull["End"], format="%d-%m-%Y %H:%M")
    ifull["WP flow activity"] = ifull["Activity"]
    pr = awt.transform_awt_to_activity_log(ifull)

    return pr



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
    return awt.group_hours(awt.compute_hourly_schedule(pr, freq='15min'), slot='15min')

def to_calendar_format(dataframe, palette=None, activities=[]):
    df = dataframe[dataframe['Activity'] != "**Empty"]
    return [{
                "title": row['Activity'] + " - " + row['Case'],
                "start": row['Begin'].isoformat(),
                "end": row['End'].isoformat(),
                "backgroundColor": palette[row['Activity']] if palette is not None else "#FFFFFF",
                "extendedProps": {
                    "activity": row['Activity'],
                    "case": row['Case']
                } 
            } for i,row in df.iterrows() if (row['Activity'] in activities if len(activities) >0 else True)]

def create_color_palette(pr):
    activities = np.append(pr["WP flow activity"].unique(),"**Misc")
    pal = sns.color_palette(palette="tab20b", n_colors=len(activities)).as_hex()
    return {a: p for a,p in zip(activities, pal)}

def create_timeline(cp, palette=None, format='%H:%M', tooltip_format=None):
    if tooltip_format is None:
        tooltip_format = format
    timeline_data = pd.DataFrame(cp[["Begin", "End"]])
    timeline_data['Begin'] = timeline_data['Begin'].dt.tz_localize('Europe/Amsterdam')
    timeline_data['End'] = timeline_data['End'].dt.tz_localize('Europe/Amsterdam')
    timeline_data['Title'] = cp["WP flow activity"] + " - " + cp["Case"]
    timeline_data['Color'] = cp["WP flow activity"].apply(lambda x: palette[x] if palette is not None else "#FFFFFF")
    timeline = alt.Chart(timeline_data).mark_bar().encode(
                    x=alt.X('Begin', axis=alt.Axis(title="", grid=True, format=format)),
                    x2=alt.X2('End', title=""),
                    y=alt.Y('Title', axis=alt.Axis(title="")),
                    color=alt.Color('Color', scale=None, legend=None),
                    tooltip=[alt.Tooltip('Begin:T', format=tooltip_format), alt.Tooltip('End:T', format=tooltip_format), 'Title']
                )
    
    return timeline

def interval_details(raw, palette, event, start, end, time_format):
    time_filter = ((raw["Begin"] >= start) & (raw["Begin"] <= end)) | ((raw["End"] >= start) & (raw["End"] <= end))
    cp = raw[time_filter][["Begin", "End", "WP flow activity", "Case"]].copy()
    cp.loc[cp["Begin"] < start, "Begin"] = start
    cp.loc[cp["End"] > end, "End"] = end
    cp["Duration"] = cp["End"] - cp["Begin"]

    main_activity = event["activity"] if event is not None and event["activity"] != "**Misc" else None

    st.subheader(f"Activities from {start.strftime(time_format)} to {end.strftime(time_format)}")
    if main_activity is not None:
        st.markdown(f"##### Main activity: {event['activity']} - {event['case']}")

    timeline = create_timeline(cp, palette)
    st.altair_chart(timeline, use_container_width=True)

    metrics = awt.compute_interval_metrics(cp, main_activity)
    display_interval_metrics(metrics, main_activity)

    with st.expander("See details of the interval:"):
        st.dataframe(cp,            
                     hide_index=True,
                     use_container_width=True,
                     column_config={
                         "Begin": st.column_config.DatetimeColumn("Begin", format="HH:mm"),
                         "End": st.column_config.DatetimeColumn("End", format="HH:mm")
                     })

def display_interval_metrics(metrics, main_activity):
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

def case_details(raw, case, palette=None):
    '''Maps the raw event log to a detailed view of the specified case
    '''
    st.subheader(f'Activities of {case}')
    
    cp = raw[raw["Case"]==case][["Begin", "End", "WP flow activity", "Case", "Duration"]].copy()

    st.markdown("##### Overview")
    st.dataframe(cp.groupby("WP flow activity", sort=False).agg(Duration=("Duration", "sum"), First=("Begin", "min"), Last=("End", "max")))
    st.multiselect("Filter by activity:", options=cp["WP flow activity"].unique(), key='case_activity_filter')
    plt.rc('font', size=12)
    if "case_activity_filter" in st.session_state and len(st.session_state.case_activity_filter) > 0:
        df = cp[cp["WP flow activity"].isin(st.session_state.case_activity_filter)]
    else:
        df = cp
    df = df.groupby(df["Begin"].dt.date)["Duration"].sum() / pd.Timedelta('1min')
    df.index = pd.to_datetime(df.index)
    fig, ax = calplot.calplot(df, suptitle="Total work in case (mins)", dropzero=True)
    st.pyplot(fig)

    metrics = awt.compute_case_metrics(raw[raw["Case"]==case], daily[daily["Case"]==case])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean duration", f'{round(metrics["MeanSlotDurationMins"],2):g} min') 
    col2.metric("Mean interruption", f'{round(metrics["MeanInterruptionDurationMins"],2):g} min')
    col3.metric("Times performed", f'{round(metrics["TimesPerformed"],2):g}')
    col4, col5, col6 = st.columns(3)
    col5.metric("Total duration", f'{round(metrics["TotalDurationHours"],2):g} hours')
    col4.metric("Interruptions per work hour", f'{round(metrics["InterruptionsPerWorkHour"],2):g}')
    col6.metric("Mean gap days", f'{round(metrics["MeanGapDays"],2):g}')

    # timeline = create_timeline(palette, cp, format=timeline_time_format, tooltip_format="%d %b %H:%M")
    # st.altair_chart(timeline, use_container_width=True)
    # data_detail = to_calendar_format(hourly[hourly["Case"]==case], palette)
    # details_event = calendar(
    #     events = data_detail, 
    #     options = {
    #         "headerToolbar": {
    #             "right": "prev,next"
    #         },
    #         "initialView": "listYear",
    #         "initialDate": "2023",
    #         "firstDay": 1
    #     },                 
    #     custom_css = custom_css,
    #     key = f'calendar_case_{case}')      


    st.markdown("##### Details")
    by_dates, by_activities = st.tabs(["By dates", "By activities"])
    with by_dates:
        #prev_day = None
        # for i, h in hourly[hourly["Case"]==case].iterrows():
        #     if prev_day is None or prev_day != h['Begin'].strftime('%d %b'):
        #         st.markdown(f"**{h['Begin'].strftime('%d %b')}**")
        #         prev_day = h['Begin'].strftime('%d %b')
        #     with st.expander(f'From {h["Begin"].strftime("%H:%M")} to {h["End"].strftime("%H:%M")}: {h["Activity"]}', expanded=True):
        #         time_filter = ((raw["Begin"] >= h["Begin"]) & (raw["Begin"] <= h["End"])) | ((raw["End"] >= h["Begin"]) & (raw["End"] <= h["End"]))
        #         cp = raw[time_filter][["Begin", "End", "WP flow activity", "Case", "Duration"]].copy()
        #         cp.loc[cp["Begin"] < h["Begin"], "Begin"] = h["Begin"]
        #         cp.loc[cp["End"] > h["End"], "End"] = h["End"]

        #         timeline = create_timeline(palette, cp)
        #         st.altair_chart(timeline, use_container_width=True)
        for name, group in raw[raw["Case"]==case].groupby(raw["Begin"].dt.date):
            with st.expander(f"**{name.strftime('%d %b')}:**", expanded=True):
                time_filter = (raw["Begin"].dt.date == name) & (raw["Case"] == case)
                df_time = raw[time_filter][["Begin", "End", "WP flow activity", "Case", "Duration"]].copy()

                timeline = create_timeline(df_time, palette)
                st.altair_chart(timeline, use_container_width=True)
    with by_activities:
        activity_metrics = awt.compute_case_metrics(raw[raw["Case"]==case], daily[daily["Case"]==case], by_activity=True)
        for name, group in raw[raw["Case"]==case].groupby("WP flow activity", sort=False):
            with st.expander(f"**{name}:**", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean duration", f'{round( activity_metrics.loc[name,"MeanSlotDurationMins"],2):g} min' )
                col2.metric("Mean interruption", f'{round(activity_metrics.loc[name,"MeanInterruptionDurationMins"],2):g} min')
                col3.metric("Times performed", f'{round(activity_metrics.loc[name,"TimesPerformed"],2):g}')
                col4, col5, col6 = st.columns(3)
                col5.metric("Total duration", f'{round(activity_metrics.loc[name,"TotalDurationHours"],2):g} hours')
                col4.metric("Interruptions per work hour", f'{round(activity_metrics.loc[name,"InterruptionsPerWorkHour"],2):g}')
                col6.metric("Mean gap days", f'{round(activity_metrics.loc[name,"MeanGapDays"],2):g}')
                for j, h in group.groupby(group["Begin"].dt.date):
                    st.markdown(f'**{j.strftime("%d %b")}**')                    
                    #st.markdown(f'**{h["Begin"].strftime("%d %b")}:** From {h["Begin"].strftime("%H:%M")} to {h["End"].strftime("%H:%M")}')
                    #time_filter = ((raw["Begin"] >= h["Begin"]) & (raw["Begin"] <= h["End"])) | ((raw["End"] >= h["Begin"]) & (raw["End"] <= h["End"]))
                    time_filter = (raw["Begin"].dt.date == j) & (raw["Case"] == case)
                    df_time = raw[time_filter][["Begin", "End", "WP flow activity", "Case", "Duration"]].copy()
                    # df_time.loc[df_time["Begin"] < h["Begin"], "Begin"] = h["Begin"]
                    # df_time.loc[df_time["End"] > h["End"], "End"] = h["End"]

                    timeline = create_timeline(df_time, palette)
                    st.altair_chart(timeline, use_container_width=True)


    with st.expander("See details of all activities:"):
        st.dataframe(
                        cp.drop("Case", axis=1),            
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Begin": st.column_config.DatetimeColumn("Begin", format="DD/MM/YY HH:mm"),
                            "End": st.column_config.DatetimeColumn("End", format="DD/MM/YY HH:mm")
                        })


def map_to_backdrop(hourly, caldf=None, palette=None, only_activities=[], include_calendar=False):
    '''Maps process (and calendar) data to the backdrop

    It uses three parameters to configure the visualization: palette to set the colors of
    the activities, only_activities to display only the activities included in the list, 
    and include_calendar to depict or not the calendar info.

    Returns
    -------
        A dict with the action performed on the calendar by the user
    '''

    if include_calendar and caldf is None:
        raise("Cannot include a calendar if it is not loaded")

    data = to_calendar_format(hourly, palette, only_activities)

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
            "height": 800,
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
        calendar_sel = calendar(events = data, options = calendar_options, callbacks=["eventClick", "select"], custom_css = custom_css, key=f"cal_with_cal_{hash(tuple(only_activities))}")
    else:
        calendar_sel = calendar(events = data, options = calendar_options, callbacks=["eventClick", "select"], custom_css = custom_css, key=f"cal_without_cal_{hash(tuple(only_activities))}")
    return calendar_sel

def change_file():
    if "raw" in st.session_state:
        del st.session_state["raw"]

st.set_page_config(layout="wide")
st.title('Calendar of activities')

with st.sidebar:
    image, text = st.columns([1,3])
    with image:
        st.image('tiramisu-banana.png')
    with text:
        st.markdown("<h1 style='text-align: center;'>Banana Tiramisù Calendar</h2>", unsafe_allow_html=True)
    #     st.text('Banana Tiramisù Calendar')
    awt_data = st.file_uploader(label='Active window tracking data', on_change=change_file)
    load_sample_data = st.button(label="Load sample data", type='primary', help="Loads sample data from  https://github.com/project-pivot/labelled-awt-data/")
    calendar_data = st.file_uploader(label='Calendar data')

if awt_data is not None or load_sample_data:  
    data_load_state = st.markdown('Loading data...')
    if awt_data is not None:
        raw = load_data(awt_data)
    elif load_sample_data:
        raw = load_sample()
    st.session_state['raw'] = raw
    print('here')
    data_load_state.text("")

if "raw" in st.session_state:
    daily = awt.compute_daily_log(st.session_state['raw'])
    palette = create_color_palette(st.session_state['raw'])
    hourly = hourly_schedule(st.session_state['raw'])
    

if calendar_data is not None:
    caldf = load_calendar(calendar_data)
else:
    caldf = None


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

if "raw" not in st.session_state:
    st.write("You need to upload the data first")
else:
    raw = st.session_state["raw"]
    filter_selection = st.multiselect(label='Filter by activity:', options=palette.keys(), key='selection')

    if calendar_data is not None:
        include_calendar = st.checkbox(label='Include calendar data', value=True)
    else:
        include_calendar = False

    col_calendar, col_details = st.columns([0.5, 0.5], gap="medium")


    with col_calendar:
        calendar_sel = map_to_backdrop(hourly, caldf, palette, filter_selection, include_calendar)        


    if "callback" in calendar_sel and calendar_sel["callback"] == "eventClick":
        event = calendar_sel["eventClick"]["event"]
        start = datetime.fromisoformat(event['start']).replace(tzinfo=None)
        end = datetime.fromisoformat(event['end']).replace(tzinfo=None)
        time_format = "%Y-%m-%d %H:%M"

        with col_details:            
            interval_detail, case_detail = st.tabs(["Interval details", "Case details"])

            with interval_detail:
                if "calendar_event" in event["extendedProps"]:
                    st.subheader(f"Calendar event: {event['title']}")
                    st.write(f"From {start.strftime(time_format)} to {end.strftime(time_format)}")
                else:
                    interval_details(raw, palette, event["extendedProps"], start, end, time_format)

            with case_detail:
                if "calendar_event" not in event["extendedProps"]:
                    if event["extendedProps"]["case"] != '**Misc':
                        case_details(raw, event["extendedProps"]["case"], palette)
                    else:
                        st.markdown("No details for misc intervals")
            
        # with col_calendar:
        #     if "activity" in event["extendedProps"]:
        #         st.write(f'Other instances of activity {event["extendedProps"]["activity"]}:')
        #             # st.write()
        #             # st.dataframe(
        #             #     hourly[hourly['Activity']==event["extendedProps"]["activity"]][["Begin", "End", "Case"]], 
        #             #     hide_index=True, 
        #             #     height=500,
        #             #     column_config={
        #             #         "Begin": st.column_config.DatetimeColumn("Begin", format="D MMM YYYY, hh:mm"),
        #             #         "End": st.column_config.DatetimeColumn("End", format="D MMM YYYY, hh:mm")
        #             #     })
                    
        #         data_detail = to_calendar_format(hourly, palette, [event["extendedProps"]["activity"]])
        #         details_event = calendar(
        #             events = data_detail, 
        #             options = {
        #                 "headerToolbar": {
        #                     "right": "prev,next"
        #                 },
        #                 "initialView": "listYear",
        #                 "initialDate": "2023",
        #                 "firstDay": 1
        #             },                 
        #             custom_css = custom_css,
        #             key = f'calendar_detail_{event["extendedProps"]["activity"]}')      

    elif "callback" in calendar_sel and calendar_sel["callback"] == "select":
        start = datetime.strptime(calendar_sel['select']['start'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=None)
        end = datetime.strptime(calendar_sel['select']['end'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=None)
        time_format = "%Y-%m-%d %H:%M"

        with col_details:            
            interval_details(raw, palette, None, start, end, time_format)


