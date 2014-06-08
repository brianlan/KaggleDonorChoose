#!/usr/bin/python

import csv
from collections import defaultdict

def projectTrainTestGenerator():
    reader = csv.reader(open("../../dataset/projects.csv", 'rU'))
    for projectid,teacher_acctid,schoolid,school_ncesid,school_latitude,school_longitude,school_city,school_state,school_zip,school_metro,school_district,school_county,school_charter,school_magnet,school_year_round,school_nlns,school_kipp,school_charter_ready_promise,teacher_prefix,teacher_teach_for_america,teacher_ny_teaching_fellow,primary_focus_subject,primary_focus_area,secondary_focus_subject,secondary_focus_area,resource_type,poverty_level,grade_level,fulfillment_labor_materials,total_price_excluding_optional_support,total_price_including_optional_support,students_reached,eligible_double_your_impact_match,eligible_almost_home_match,date_posted in reader:
        if projectid != 'projectid':
            yield projectid, date_posted

def projectExcitingGenerator():
    reader = csv.reader(open("../../dataset/outcomes.csv", 'rU'))
    for projectid,is_exciting,at_least_1_teacher_referred_donor,fully_funded,at_least_1_green_donation,great_chat,three_or_more_non_teacher_referred_donors,one_non_teacher_referred_donor_giving_100_plus,donation_from_thoughtful_donor,great_messages_proportion,teacher_referred_count,non_teacher_referred_count in reader:
        if projectid != 'projectid':
            yield projectid, is_exciting

def projectEssayGenerator():
    reader = csv.reader(open("../../dataset/essays.csv", 'rU'))
    for projectid, teacher_acctid, title, short_description, need_statement, essay in reader:
        if projectid != 'projectid':
            yield projectid, essay

# split training and testing data
project_train_test = []
for projectid, date_posted in projectTrainTestGenerator():
    if date_posted < '2014-01-01':
        project_train_test.append([projectid, 'train'])
    else:
        project_train_test.append([projectid, 'test'])

# extract the essay
project_essay = defaultdict()
for projectid, essay in projectEssayGenerator():
    project_essay[projectid] = essay

# get the labels (is_exciting)
project_label = {}
for projectid, is_exciting in projectExcitingGenerator():
    project_label[projectid] = 0 if is_exciting == 'f' else 1

project_data = []
for proj in project_train_test:
    if project_label.has_key(proj[0]):
        label = project_label[proj[0]]
    else:
        label = 'u'

    if project_essay.has_key(proj[0]):
        essay = project_essay[proj[0]]
    else:
        essay = ''

    project_data.append([proj[0], essay, proj[1], label])

writer = csv.writer(open('../../dataset/project_integrated_data.csv', 'wb'), delimiter=',')
writer.writerow(['projectid', 'essay', 'train_test', 'is_exciting'])
writer.writerows(project_data)
