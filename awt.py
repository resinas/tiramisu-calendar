import pandas as pd
import numpy as np

def transform_awt_to_activity_log(dataframe, inactivity_threshold=pd.Timedelta("1m")):
    """Transforms an active window tracking log into an activity log

    To transform it, we group all active windows events until a change happens. We consider it a change if the activity or the case assigned
    to the active window is different or if the inactivity period between two active windows (the difference between the end of one event
    and the beginning of the next one) is above the inactivity_threshold. At this moment we only keep the first window title.

    Together with the transformation, we add several additional columns that represent the duration of each activity (Duration), the same
    duration in minutes (Duration_minutes), the previous activity performed in each day (Prev), whether there is a gap in the recorded log
    before the current activity (Gap) and the interruption time since the previous activity performed (interruption_time). 

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
    change = ((dataframe["WP flow activity"].shift() != dataframe["WP flow activity"]) | (dataframe["Case"].shift() != dataframe["Case"])  |  ((dataframe["Begin"] - dataframe["End"].shift()) > inactivity_threshold))
    it = change.cumsum()

    pr = dataframe.groupby(by=it).agg({"Begin": "first", "End": "last", "WP flow activity": "first", "Case":"first", "Title": "first"})
    pr["Duration"] = pr["End"] - pr["Begin"]
    pr["Begin"] = pd.to_datetime(pr["Begin"])
    pr["End"] = pd.to_datetime(pr["End"])
    pr["Duration_minutes"] = pr["Duration"] / pd.Timedelta('1m')
    prev = pr["WP flow activity"].shift()
    prev.loc[pr["Begin"].dt.date != pr["Begin"].shift().dt.date] = np.nan
    pr["Prev"] = prev
    # We consider the gap only within the same day (alternatively, we could also consider there is a gap before the first activity in the morning)
    pr["Gap"] = ((pr["Begin"] - pr["End"].shift() > inactivity_threshold) & (pr["Begin"].dt.day == pr["End"].shift().dt.day))    

    # This row represents a within-day interruption, which is the 
    interruption_time_b = pr.groupby(["WP flow activity", "Case"]).apply(lambda x: x["Begin"]- x["End"].shift())
    interruption_time_b.loc[pr.groupby(["WP flow activity", "Case"]).apply(lambda x: x["Begin"].dt.day != x["End"].shift().dt.day)] = np.nan
    pr["interruption_time"] = interruption_time_b.reset_index(level=[0,1], drop=True) / pd.Timedelta('1m')

    return pr

def compute_daily_log(dataframe):
    """Computes a daily log of activities and cases performed from a dataframe with an activity log

    The computed daily log includes for each activity, case and day, the duration of the activity and
    case each day (Duration), the number of times performed during the day (Times), and the number of
    days since it was last performed (Gap). The Gap information is also included by case (CaseGap).

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe with the activity event log 

    Returns
    -------
    dataframe
        The dataframe with the daily log
    """

    pdaily = dataframe.groupby(["WP flow activity", "Case", dataframe["Begin"].dt.date]).agg(Duration=('Duration', "sum"), Times=('Duration', "count")).reset_index().sort_values(["WP flow activity", "Case", "Begin"])
    pdaily["Duration"] = pdaily["Duration"] / pd.Timedelta('1h')
    pdaily["Begin"] = pd.to_datetime(pdaily["Begin"])
    pdaily["Gap"] = pdaily.groupby(['WP flow activity', 'Case'])["Begin"].transform(lambda x: (x - x.shift())/pd.Timedelta('1d')).fillna(0)
    pdaily["CaseGap"] = pdaily.groupby('Case')["Begin"].transform(lambda x: (x - x.shift())/pd.Timedelta('1d')).fillna(0)
    # pdaily["CaseDuration"] = pdaily.groupby(['Case', 'Begin'])["Duration"].transform("sum")
    # pdaily["CaseTimes"] = pdaily.groupby(['Case', 'Begin'])["Times"].transform("sum")

    return pdaily


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


def compute_interval_metrics(dataframe, activity=None):
    """Compute metrics for an interval provided in dataframe

    Parameters
    ----------
    dataframe : DataFrame
        A dataframe with an interval of an activity log for which the metrics are computed
    activity: str, optional
        The main activity of the interval that is used as the reference to compute the metrics.
        If no main activity is provided, the metrics are computed considering all activities as
        equal.

    Returns
    -------
    dict
        A dictionary with the metrics computed for the interval.
    """

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


def compute_case_metrics(activity_log, daily_log, by_activity=False, freq = None, long_interruption=60):
    if activity_log["Case"].nunique() > 1 or daily_log["Case"].nunique() > 1:
        raise("Both the activity log and daily log must refer to just one case")
    
    groupby_spec = ["WP flow activity"] if by_activity else []
    if freq is not None:
        groupby_spec = groupby_spec + [pd.Grouper(key="Begin", freq=freq)]

    daily_log["NumInterr"] = daily_log["Times"] - 1

    if len(groupby_spec) > 0:
        proj = activity_log.groupby(groupby_spec)
        pdaily = daily_log.groupby(groupby_spec)        
    else:
        proj = activity_log
        pdaily = daily_log

    g = proj.agg(MeanSlotDurationMins=("Duration_minutes", "mean"), MeanInterruptionDurationMins=("interruption_time", "mean"))


    d = pdaily.agg(TimesPerformed=("Times", "sum"), NumInterr=("NumInterr", "sum"), TotalDurationHours=("Duration", "sum"), MeanGapDays=("CaseGap", "mean"))

    if len(groupby_spec) == 0:
        g = g.fillna(0).sum(axis=1)
        d = d.fillna(0).sum(axis=1)

    d["InterruptionsPerWorkHour"] = (d["NumInterr"]) / d["TotalDurationHours"]

    daily_log.drop("NumInterr", axis=1, inplace=True)

    if len(groupby_spec) == 0:
        return pd.concat([g, d], axis=0).fillna(0)
    else:
        return pd.concat([g, d], axis=1).fillna(0)   
    
    
    # if freq is not None:
    #     g2 = activity_log[activity_log["interruption_time"] < long_interruption].groupby(by=["WP flow activity", pd.Grouper(key="Begin", freq=freq)])["interruption_time"].agg("mean").rename("MeanInterruptionDurationNoOutliersMins")
    #     g3 = activity_log[activity_log["interruption_time"] >= long_interruption].groupby(["WP flow activity", pd.Grouper(key="Begin", freq=freq)])["interruption_time"].agg("count").rename("LongInterruptionTimesNum")
    #     num_activities = daily_log.groupby(pd.Grouper(key="Begin", freq=freq))["WP flow activity"].nunique().rename("NumDifferentActivities")
    #     return pd.merge(pd.concat([g, g2, g3, d], axis=1), num_activities, left_on="Begin", right_index=True, how="left")
    # else:
    #     g2 = activity_log[activity_log["interruption_time"] < long_interruption].groupby(by="WP flow activity")["interruption_time"].agg("mean").rename("MeanInterruptionDurationNoOutliersMins")
    #     g3 = activity_log[activity_log["interruption_time"] >= long_interruption].groupby("WP flow activity")["interruption_time"].agg("count").rename("LongInterruptionTimesNum")
    #     num_activities = daily_log["WP flow activity"].nunique().rename("NumDifferentActivities")
    #     return pd.concat([g, g2, g3, d], axis=1)