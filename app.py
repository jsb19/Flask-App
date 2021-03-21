#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:25:33 2021

@author: jujharbedi
"""
# Importing flask framework to deploy project
from flask import Flask, render_template, request, redirect, url_for, session, Response
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# Importing libraries for data analaysis
import random
import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Import data
dataset = pd.read_csv('./static/data/shot_data.csv')

# Placeholder for raw dataset
raw = dataset
raw.head()

# All column headers in dataset
data_columns = ["action_type	", "combined_shot_type",	"game_event_id",	"game_id", 
                "lat","loc_x", "loc_y" ,"lon", "minutes_remaining", "period",
                "playoffs", "season", "seconds_remaining", "shot_distance", 
                "shot_made_flag","shot_type", "shot_zone_area", "shot_zone_basic",
                "shot_zone_range", "team_id", "team_name", "game_date",
                "matchup", "opponent", "shot_id"]

# Column headers we want to keep
keep_columns = ["action_type	", "combined_shot_type",	"game_event_id",	"game_id", 
                "lat", "loc_x", "loc_y" ,"lon", "period", "playoffs", "season", 
                "shot_distance", "shot_made_flag", "shot_zone_area", "shot_zone_basic", "shot_zone_range", "opponent",
                "shot_id"]

for i in range(len(data_columns)):
    if not data_columns[i] in keep_columns:
        dataset.drop(data_columns[i], inplace = True, axis = 1)
        
# Creating temp variables for plotting
datapoints = dataset.values

#Removing nan values for shot_flag (data cleaning)
dataset = dataset[pd.notnull(dataset['shot_made_flag'])]

##------ Got this function for drawing a basketball court from here: http://savvastjortjoglou.com/nba-shot-sharts.html ------#
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


# Creating a function to plot x and y coordinates based on shot zones
def scatter_plot_by_category(plot, feat):
    alpha = 0.1
    # Grouping data frame by shot category
    gs = dataset.groupby(feat)
    # Creating color map for each category
    cmap = cm.get_cmap('viridis')
    # Create RBG values for each color
    colors = cmap(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, colors):
        plot.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha,  label=g[0])
    plot.legend(markerscale=2)


# Creating figure for Shots by Shot Zone
figure1 = plt.figure(1, figsize=(20,10))

# Subplot for shots by shot_zone_area
plt1 = plt.subplot(131)
scatter_plot_by_category(plt1, 'shot_zone_area')
plt.title('Shots by shot_zone_area', fontsize=20)
draw_court(outer_lines=True)

# Subplot for shots by shot_zone_basic
plt2 = plt.subplot(132)
scatter_plot_by_category(plt2, 'shot_zone_basic')
plt.title('Shots by shot_zone_basic', fontsize=20)
draw_court(outer_lines=True)

# Subplot for shots by shot_zone_range
plt3 = plt.subplot(133)
scatter_plot_by_category(plt3, 'shot_zone_range')
plt.title('Shots by shot_zone_range', fontsize=20)
draw_court(outer_lines=True)

# Saving figure as a image file 
figure1.savefig('./static/img/shotZonePlot.png')


# Plotting Made vs Missed Shots by Year
data_made = dataset.shot_made_flag == 1
data_missed = dataset.shot_made_flag == 0
shot_missed = dataset[data_missed].season.value_counts()
shot_success = dataset[data_made].season.value_counts()
shots = pd.concat([shot_success,shot_missed],axis=1)
shots.columns=['Success','Missed']
figure2 = plt.figure(2, figsize=(22,9))
shots.plot(ax=figure2.add_subplot(111), kind='bar',stacked=False,rot=1,color=['#008000','#FF0000'])
plt.xlabel('Season')
plt.ylabel('Number of shots')
plt.legend(fontsize=15)
plt.title("Made vs Missed Shots by Year", fontsize=20) 
# Saving figure as a image file 
figure2.savefig('./static/img/madeVsMissedShots.png')


def getPieChart(i, zone):

    # Plotting number of shots taken per shot range
    pieData = dataset[zone].value_counts()
    
    # Checking which zone
    if "range" in zone:
        # Removing all shots from beyond half court 
        pieData = pieData.drop("Back Court Shot")
        title = "Shots Made by Distance"
        filePath = "./static/img/pieChartDistance.png"
    if "area" in zone:
        # Removing all shots from beyond half court 
        pieData = pieData.drop("Back Court(BC)")
        title = "Shots Made by Location"
        filePath = "./static/img/pieChartLocation.png"
    if "basic" in zone:
        # Removing all shots from beyond half court 
        pieData = pieData.drop("Backcourt")
        title = "Shots Made by Type"
        filePath = "./static/img/pieChartBasic.png"

    figure3 = plt.figure(i,  figsize=[11,8])
    # Getting labels for pie chart
    labels = pieData.keys()
    numLabels = len(labels)
    plt.pie(x=pieData, autopct="%.1f%%", explode=[0.05]*numLabels, labels=labels, pctdistance=.5)
    plt.title(title, fontsize=20)
    # Saving figure as a image file 
    figure3.savefig(filePath)

# Iterate over shotzones to get pieChart of each shot zone category
shotZones = ["shot_zone_range", "shot_zone_area", "shot_zone_basic"]
# Counter for figures to prevent duplicates
i = 5
for zone in shotZones:
    i = i + 1
    getPieChart(i, zone)

# Plotting percentage made per shot range
def plotAccuracyByZone(zone):

    # Create dataset with shot zone and 
    data = dataset[[zone, "shot_made_flag"]]
    # Getting number of shots made and missed
    data = data.groupby(by=[zone, "shot_made_flag"]).size()
    # Creating placeholders for bar chart columns
    zones = []
    made_shots = []
    missed_shots = []
    for index, count in enumerate(data):
        # Remove Backourt Shots from data
        if "Backcourt" in data.keys()[index][0] or "Back Court" in data.keys()[index][0]:
            continue
        # Prevent duplicate zones from being added
        if index % 2 == 0:
            zones.append(data.keys()[index][0])
        # Creating array of made shots by zone
        if data.keys()[index][1] == 1.0:
            made_shots.append(count)
        # Creating array of missed shots by zone
        else:
            missed_shots.append(count)

    if "range" in zone:
        title = "Shots Made by Distance"
        filePath = "./static/img/distance.png"
    if "area" in zone:
        title = "Shots Made by Location"
        filePath = "./static/img/location.png"
    if "basic" in zone:
        title = "Shots Made by Type"
        filePath = "./static/img/basic.png"

    figure = plt.figure(figsize=[11,8])

    # Width of bar chart
    width = 0.35
    plt.bar(zones, made_shots, width, label='Shots Made', color="green")
    plt.bar(zones, missed_shots, width, bottom=made_shots, label='Shots Missed', color="red")

    plt.ylabel("Shot Attempts")
    plt.title(title, fontsize=20)
    plt.legend(loc='upper center')
    # Make space for and rotate the x-axis tick labels
    figure.autofmt_xdate()
    # Saving figure as a image file 
    figure.savefig(filePath)
    # Clear figure
    plt.clf()

    return filePath


# DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
executor = ThreadPoolExecutor(1)

@app.route("/")
def home():

    # Load extensive prediction.py file in background
    executor.submit(loadFile)

    # Create secondary data frame to output as table in order to show some sample data
    table = raw.head(30)

    # Get selection for stacked bar chart
    zone = request.args.get('jsdata')
    if zone:
        # Plot accuracy for selected Zone
        path = plotAccuracyByZone(zone)
        return path
    else:
        # Show default chart
        path = plotAccuracyByZone("shot_zone_range")
    
    return render_template("index.html", tables=[table.to_html(index = False, classes='table table-bordered table-striped table-hover', header="true")], titles=table.columns.values, chart = path)


@app.route("/prediction")
def prediction():

    if request.method == "GET":
        # Import data from prediction file
        from prediction import dataset, featureImportance, newFeatureImportance, rawPredData, predData, accuracy

        encodedData = dataset.head(30)

        rawPredData = rawPredData.head(50)

        predData = predData.head(50)

        return render_template("prediction.html", encodedData=[encodedData.to_html(index = False, classes='table table-bordered table-striped table-hover')], featureImp=[featureImportance.to_html(index = False, classes='table table-bordered table-striped table-hover')], newFeatureImp=[newFeatureImportance.to_html(index = False, classes='table table-bordered table-striped table-hover')],
        pred=[rawPredData.to_html(index = False, classes='table table-bordered table-striped table-hover')], roundedPred=[predData.to_html(index = False, classes='table table-bordered table-striped table-hover')], result=accuracy)


def loadFile():
    from prediction import accuracy
    sleep(10)