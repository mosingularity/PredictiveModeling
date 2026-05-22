# Databricks notebook source
#from datetime import date
import datetime
import holidays
import json
from dateutil.relativedelta import relativedelta

days = []
months = int(dbutils.widgets.get("months"))

dateStart = datetime.date.today()
dateEnd = datetime.date.today() + relativedelta(months=months)
yearStart = dateStart.year
yearEnd = dateEnd.year + 1

print(yearStart, yearEnd)

sa_holidays = holidays.country_holidays('ZA')

for date, name in sorted(holidays.ZA(years=range(yearStart,yearEnd)).items()):
  if date >= dateStart and date <= dateEnd:
    days.append("'" + date.strftime('%Y-%m-%d') + "'")
print (days)

dbutils.notebook.exit(json.dumps(days))