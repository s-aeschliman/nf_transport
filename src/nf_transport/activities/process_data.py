import pandas as pd
from torch.utils.data import Dataset


class Activity:
    def __init__(self, start, end, act_type, loc):
        self.start = start
        self.end = end
        self.type = act_type
        self.loc = loc
        self.duration = self.end - self.start

    def __str__(self):
        return f"{self.type} activity starting at {self.start} and ending at {self.end}"


class Trip:
    def __init__(self, start, end, travtime):
        self.start = start
        self.end = end
        self.duration = travtime

    def __str__(self):
        return f"trip lasting {self.duration} minutes."


class Schedule:
    def __init__(self, activities: list[Activity], trips: list[Trip]):
        self.activities = activities
        self.trips = trips

        # build the schedule. sort by start times to alternate activities and trips
        schedule = activities + trips
        self.schedule = sorted(schedule, key=lambda x: x.start)

    def __str__(self):
        text = "-----------------------\n"
        for a in self.schedule:
            text = text + a.__str__() + "\n"
        text = text + "-----------------------\n"

        return text


class Person:
    def __init__(self, id: str, household: str):
        self.id = id
        self.household = household
        self.schedule = None

    def set_schedule(self, schedule: Schedule):
        self.schedule = schedule


class Household:
    def __init__(self, id: str, personlist: list[Person]):
        self.id = id
        self.personlist = personlist


class ScheduleDataset(Dataset):
    def __init__(self, schedules: list[Schedule]):
        self.schedules = schedules

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        return self.schedules[idx]


def process_cmap():
    person = pd.read_csv("data/cmap/person.csv")
    household = pd.read_csv("data/cmap/household.csv")
    location = pd.read_csv("data/cmap/location.csv")
    place = pd.read_csv("data/cmap/place.csv")

    person["id"] = person["sampno"].astype("str") + person["perno"].astype("str")
    place["id"] = place["sampno"].astype("str") + place["perno"].astype("str")

    return person, place


def create_schedules():

    person, place = process_cmap()
    place["arrtime"] = pd.to_datetime(place["arrtime"])
    place["deptime"] = pd.to_datetime(place["deptime"])

    schedules = []
    people = []

    # loop for now for clarity, speed up later
    for i, p in person.groupby("id"):
        person_trips = place[place.id == i].copy()
        person_trips["next_arrtime"] = person_trips["arrtime"].shift(-1)
        person_trips["next_travtime"] = person_trips["travtime"].shift(-1)

        p_ = Person(id=str(i), household=str(p.sampno.unique()))

        activities = []
        trips = []
        for i, r in person_trips.iterrows():
            a = Activity(
                start=r["arrtime"],
                end=r["deptime"],
                act_type="placeholder",
                loc=str(r["locno"]),
            )
            activities.append(a)

            if pd.notna(r["next_arrtime"]):
                t = Trip(
                    start=r["deptime"],
                    end=r["next_arrtime"],
                    travtime=r["next_travtime"],
                )
                trips.append(t)

        s = Schedule(activities=activities, trips=trips)
        p_.set_schedule(s)
        schedules.append(s)
        people.append(p)
        print(s)

    # schedule_data = ScheduleDataset(schedules=schedules)

    return schedules, people


create_schedules()
