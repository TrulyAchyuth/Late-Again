# EDA summary (synthetic data)

Trips: 6,000  |  Mean delay: 7.39 min  |  Delayed (> threshold): 53.2%


## By transport_type

| transport_type   |   trips |   mean_delay_min |   pct_delayed |
|:-----------------|--------:|-----------------:|--------------:|
| Bus              |    2500 |             9.27 |          63.2 |
| Train            |    1448 |             7.56 |          57.7 |
| Tram             |    1233 |             6.16 |          46   |
| Metro            |     819 |             3.21 |          25.5 |



## By weather_condition

| weather_condition   |   trips |   mean_delay_min |   pct_delayed |
|:--------------------|--------:|-----------------:|--------------:|
| Storm               |     352 |            18.52 |          96   |
| Heavy Rain          |     593 |            13.5  |          90.6 |
| Snow                |     249 |            12.59 |          88.8 |
| Rain                |     769 |             8.07 |          60.6 |
| Fog                 |     467 |             7.85 |          61.2 |
| Cloudy              |    1336 |             5.19 |          41.6 |
| Clear               |    2234 |             4.42 |          35.2 |



## By event_type

| event_type   |   trips |   mean_delay_min |   pct_delayed |
|:-------------|--------:|-----------------:|--------------:|
| Sports       |     142 |            14.89 |          93   |
| Concert      |     140 |            13.51 |          85.7 |
| Parade       |      63 |            11    |          74.6 |
| Festival     |     145 |             9.81 |          72.4 |
| Protest      |     122 |             7.3  |          57.4 |
| No Event     |    5388 |             6.93 |          50.4 |



## By is_peak_hour

|   is_peak_hour |   trips |   mean_delay_min |   pct_delayed |
|---------------:|--------:|-----------------:|--------------:|
|              1 |    2077 |             9.98 |          72.7 |
|              0 |    3923 |             6.02 |          42.9 |



## By is_weekend

|   is_weekend |   trips |   mean_delay_min |   pct_delayed |
|-------------:|--------:|-----------------:|--------------:|
|            0 |    4250 |             7.95 |          57.3 |
|            1 |    1750 |             6.03 |          43.1 |



## By is_holiday

|   is_holiday |   trips |   mean_delay_min |   pct_delayed |
|-------------:|--------:|-----------------:|--------------:|
|            0 |    5818 |             7.53 |          54.1 |
|            1 |     182 |             3.02 |          23.6 |

