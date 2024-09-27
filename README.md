# Tiramisù calendar

Tiramisù calendar is a web application designed to provide insights about personal work processes. For instance, for an academic, her personal work processes performed during her daily work might involve writing a paper, preparing a course, reviewing research papers, or supervising a PhD student, among many other processes. 

Tiramisù calendar follows the [Tiramisù conceptual framework](https://doi.org/10.1007/s10844-024-00875-8), which is a conceptual framework aimed at designing visualizations for knowledge-intensive and unstructured processes, which are characteristics that can be frequently found in personal work processes. 

You can try Tiramisù calendar at [https://tiramisu-calendar.streamlit.app](https://tiramisu-calendar.streamlit.app).

## Input data

To analyze personal work processes, the tool must be provided with collected personal information in the form of an event log that can be analyzed using process mining techniques. There are several techniques that can be used to collect personal information such as timesheet techniques or screen recordings, each with their own advantages and disadvantages. Tiramisù calendar supports two types of data:

### Active Window Tracking data
Active Window Tracking data can be seen as an event log that records the active window in a computer at any moment in time. Each event includes the title of the active window, the name of the corresponding app, and the timestamps when the window became active and stopped being active. There are several different tools that record this information. One of them is [Tockler](https://maygo.github.io/tockler/), which is open source and stores all data locally, avoiding privacy concerns. We also assume that the user has labelled each event in the log with information about:
1. The activity that was being done at that moment, e.g., conducting research, preparing lectures, etc.
2. The case the activity belongs to, e.g., a research paper. 
The [Worktagger tool](https://github.com/project-pivot/worktagger) can be used to support the user in the labelling task. 

### Calendar data
Calendar data can be obtained from any calendar management application and it involves the events scheduled in one's calendar including the start and end time, and the title of the event.

## The tool

Active Window Tracking data is visualized on the left-hand side on top of the calendar backdrop. The activities are represented by boxes, titled as: [activity name] - [case name]. The position and size of each box represents when the activity started and its duration, respectively. The buttons above the calendar allow the user to change the calendar view to depict one day, one month, or a list view of all the activities performed like it usually happens in typical calendar tools.
The checkbox labelled *Include calendar data* allows the user to show or hide the calendar data layer on top of the backdrop. The Active Window Tracking data cannot be hidden, but it is possible to show only the activities selected in the *Filter by activity* drop box that appears in the top of the figure. 

In many cases, for instance, when a person multitasks or when there are frequent interruptions in the environment, the duration of activities can be very short, e.g., around two or three minutes. Instead, calendars work best with a granularity of at most 15 minutes. Therefore, representing those fine-grained activities would make the calendar too cluttered, and it would negatively impact the quality of the visualization. For this reason, we abstract the activities obtained from the Active Window Tracking data to 15-minute time slots. Specifically, we assign each 15-minute time slot to the activity that has been performed for the longest time in that period as long as it is above a certain threshold. Otherwise, the 15-minute time slot is assigned to a *misc* activity or to no activity if the user has been inactive for the majority of the time slot. 

Besides the overview provided by the calendar and the layers of information on top of it, the tool is designed to work following the details-on-demand principle. Specifically, when the user clicks in any of the intervals in the calendar, a full set of details and metrics regarding that interval appears, as shown in the right-hand side of the screenshot. The information provided includes:
1. The details of the activities performed in the interval that were abstracted away in the calendar.
2. Some metrics about interruptions, effectiveness, and other activities performed in the same interval.
3. The details of the events and windows active in the interval. 
Similarly, by clicking in *Case details* the user can find details of the case performed in the interval selected. 

## Project setup

After cloning the repository, install all dependencies with:

```
pip3 install -r requirements.txt
```

Once all dependencies have been installed, you can run it with:

```
streamlit run tiramisu-calendar.py
```

You can access the application at [http://localhost:8501](http://localhost:8501).

## Run using docker

Alternatively, you can create a Docker image:

```
docker build -t tiramisucalendar:latest .
```

And then run the image with:

```
docker run --rm -d -p 8501:8501/tcp tiramisucalendar:latest
```

You can access the application at [http://localhost:8501](http://localhost:8501).