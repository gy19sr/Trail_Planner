from openai import OpenAI
import streamlit as st
import folium
from streamlit_folium import st_folium
import getpass
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import folium
from typing import Annotated, List, Tuple, Union
from streamlit_folium import folium_static
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import base64
import pandas as pd
from math import cos, radians
import geopandas as gpd
from shapely.geometry import Point
import chromadb
from folium.features import DivIcon

# UserWarning: Importing verbose from langchain root module is no longer supported. Please use langchain.globals.set_verbose() / langchain.globals.get_verbose() instead.
            
# add image in top left 

# https://github.com/AdieLaine/Streamly/tree/main


# environment, this may be different in streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
langchain_tracing = st.secrets["LANGCHAIN_TRACING_V2"]
langchain_project = st.secrets["LANGCHAIN_PROJECT"]

# model
model_filter = ChatOpenAI(model="gpt-4o", temperature=0)
chroma_client = chromadb.Client()

### tools

@tool
class is_this_topic_referenced(BaseModel):
    """Your job is to determine if each of the following topics were mentioned in the input query abput going on a hiking trip.
     If you're ever unsure or it doesn't feel implicit assume not mentioned."""

    location: bool = Field(..., description="did the text provide a location? if unsure say false")
    travel_days_hours_distance: bool = Field(..., description="""did they a reference to number of days/ hours they wanted to go for (driving)? 
    or how far out from their location they are willing to travel? if unsure say false""")
    hiking_distance_or_time: bool = Field(..., description="did they reference to how long they want the hikes to be in hours or miles? if unsure say false")
    number_of_hikes: bool = Field(..., description="did the user mention how exactly many hikes they wanted to do? if unsure say false")
    hiking_difficulty: bool = Field(..., description="did the user mention how difficult they wanted their hike to be? easy, medium, and hard are the options. if unsure say false")


@tool("travel_dist_tool")
class travel_dist_tool(BaseModel):
    """Your job is to determine how far people are willing to travel for a hike. be carefull, you may also be given information about hike distance, you need to understand how far they are willing to travel from a starting location!
    extract the number and metric from the text either miles, kilometers, hours or days"""

    metric: str = Field(..., description="for travel distance did the user provide hours, days, miles or kilometers?")
    value: float = Field(..., description="what was the number value what they provided?")

@tool("start_location_tool")
class start_location_tool(BaseModel):
    """extract the starting location from the user and convert to a lat and long output"""

    Location_name: str = Field(description="The name of the location the user mentioned")
    Latitude: float = Field(description="The Latitude of the location mentioned")
    Longitude: float = Field(description="The Longitude of the location mentioned")

@tool("hiking_distance_tool")
class hiking_distance_tool(BaseModel):
    """extract the distance the user is willing to hike or walk"""

    hike_metric: str = Field(description="did they say hours or miles?")
    hike_distance: float = Field(description="The number of hours or miles they wanted to hike")
    
@tool("number_of_hikes_tool")
class number_of_hikes_tool(BaseModel):
    """extract the number of trails, hikes or walks the user wants to do """

    num_hikes: float = Field(description="The number of hikes the user said they want to do")

@tool("hike_difficulty_tool")
class hike_difficulty_tool(BaseModel):
    """Determine what difficulty the user is looking for for their hikes.
    select only one from the list: [easy, medium, hard]. if unsure choose medium"""

    hike_difficulty: str = Field(description="The difficulty of the hike select only one from the list: easy, medium, hard")

# "gpt-3.5-turbo-0125"
topics_mentioned_llm = model_filter.with_structured_output(is_this_topic_referenced)
travel_dist_tool_llm = model_filter.with_structured_output(travel_dist_tool)
start_location_tool_llm = model_filter.with_structured_output(start_location_tool)
hiking_distance_tool_llm = model_filter.with_structured_output(hiking_distance_tool)
number_of_hikes_tool_llm = model_filter.with_structured_output(number_of_hikes_tool)
hike_difficulty_tool_llm = model_filter.with_structured_output(hike_difficulty_tool)

# Function to calculate bounds of a circle
def get_circle_bounds(center, radius):
    lat, lon = center
    delta_lat = radius / 111320  # Approximate conversion from meters to degrees latitude
    delta_lon = radius / (40075000 * cos(radians(lat)) / 360)  # Approximate conversion from meters to degrees longitude
    bounds = [
        [lat - delta_lat, lon - delta_lon],
        [lat + delta_lat, lon + delta_lon]
    ]
    return bounds

# define filtering functions for trails
def filter_trails_for_travel_distance(trails, start_location, dist_metric, dist_value):

    latitude = start_location['Latitude']
    longitude = start_location['Longitude']

    if dist_metric == 'days':
        lat_conversion = 111  # kilometers per degree latitude
        lon_conversion = 111 * abs(cos(radians(latitude)))  # kilometers per degree longitude based on latitude
        result = dist_value * 65  / ((lat_conversion + lon_conversion) / 2)  # Convert to degrees
    
    elif dist_metric == 'hours':
        lat_conversion = 111  # kilometers per degree latitude
        lon_conversion = 111 * abs(cos(radians(latitude)))  # kilometers per degree longitude based on latitude
        result = dist_value * 20  / ((lat_conversion + lon_conversion) / 2)  # Convert to degrees

    elif dist_metric == 'miles':
        kilo = dist_value * 1.60934 # Convert miles to kilometers first
        lat_conversion = 111  # kilometers per degree latitude
        lon_conversion = 111 * abs(cos(radians(latitude)))  # kilometers per degree longitude based on latitude
        result = kilo / ((lat_conversion + lon_conversion) / 2) 

    elif dist_metric == 'kilometers':
        lat_conversion = 111  # kilometers per degree latitude
        lon_conversion = 111 * abs(cos(radians(latitude)))  # kilometers per degree longitude based on latitude
        result = dist_value / ((lat_conversion + lon_conversion) / 2)  # Convert to degrees
    else:
        result = 100 * 1.60934 / 110 # Convert to degrees

    degree_buffer = result

    # Convert to a shapely point
    center_point = Point(longitude, latitude)

    # Create a buffer around the point
    buffer = center_point.buffer(degree_buffer)

    # Assuming 'trails' is your GeoDataFrame
    # Filter trails that intersect with the buffer
    filtered_trails = trails[trails.geometry.intersects(buffer)]

    return filtered_trails, degree_buffer

def filter_trails_for_hiking_distance(trails, hike_metric, hike_distance):

    if hike_metric == 'miles':
        filtered_trails = trails[trails['lengthmile'] >= hike_distance - 1.5]
        filtered_trails = filtered_trails[filtered_trails['lengthmile'] <= hike_distance + 1.5]
       # print('hiking distance: ', filtered_trails)
        return filtered_trails
    if hike_metric == 'hours':
       # print(trails.columns)
        filtered_trails = trails[trails['completion_time_hours'] >= hike_distance - 1]
        filtered_trails = filtered_trails[filtered_trails['completion_time_hours'] <= hike_distance + 1]
        return filtered_trails

#num_hikes
def filter_trails_for_num_hikes(trails, number_of_hikes):
    filtered_trails = trails.head(number_of_hikes)
    return filtered_trails

#num_hikes
def filter_trails_for_difficulty(trails, difficulty):
    filtered_trails = trails[trails['difficulty'] == difficulty ]
    return filtered_trails

# format markdown output
def format_output(data, title):
    formatted = [f"### {title}"]
    for key, value in data.items():
        formatted_key = key.replace('_', ' ').capitalize()
        formatted.append(f"- **{formatted_key}**: {value}")
    return "\n".join(formatted)

# Function to convert degrees to meters
def degrees_to_meters(lat, lon, radius_degrees):
    # Approximate conversions
    lat_conversion = 111000  # meters per degree latitude
    lon_conversion = 111000 * abs(cos(lat * (3.141592653589793 / 180)))  # meters per degree longitude based on latitude
    return radius_degrees * ((lat_conversion + lon_conversion) / 2)

# Define a style function for the white outline
def outline_style_function(feature):
    return {
        'color': 'white',
        'weight': 7,  # Outline thickness
        'opacity': 1
    }

# Define a style function for the green trail
def trail_style_function(feature):
    return {
        'color': 'green',
        'weight': 5,  # Main line thickness
        'opacity': 1
    }

# Define a highlight function for the trails to create a white outline effect
def highlight_function(feature):
    return {
        'color': 'white',
        'weight': 8,  # Outline thickness
        'opacity': 1
    }

# Function to get the start coordinate
def get_start_location(geom):
    if geom.geom_type == 'LineString':
        return list(geom.coords[0])
    elif geom.geom_type == 'MultiLineString':
        # Get the start coordinate of the first LineString in the MultiLineString
        return list(geom.geoms[0].coords)[0]
    else:
        return None


# Function to convert tuples to lists
def convert_to_list(x):
    if isinstance(x, tuple):
        return list(x)
    else:
        return x

# Function to invert the order of elements in lists
def invert_list_order(x):
    if isinstance(x, list) and len(x) == 2:
        return [x[1], x[0]]  # Invert the order of elements in the list
    else:
        return x

# Apply the function to the column

##############################################################################################
# start of streamlit
##############################################################################################

# import trails
# Load the GeoDataFrame only once and store it in session state
if "trails" not in st.session_state:
    # Apply the function to the geometry column
   
    st.session_state.trails = gpd.read_file("trails_cali_describe_full.geojson")
    
    print("trails loaded")
    # can load trails after the request and map plot 
    #trails = st.session_state.trails


st.title("California Trail Trip Planner")

if "topics_mentioned" not in st.session_state:
    st.session_state.topics_mentioned = []

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=openai_api_key)

# Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and display sidebar image with glowing effect
img_path = "trail_planner_image.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow" width="300" height="300">',
    unsafe_allow_html=True,
)

 # Toggle checkbox in the sidebar for basic interactions
show_basic_info = st.sidebar.toggle("Show Basic Interactions", value=True)


# Using "with" notation
with st.sidebar:

    # I can put in some text up here to start things out
    st.sidebar.markdown("---")

    # create a session variable for the model
   # if "openai_model" not in st.session_state:
    #    st.session_state["openai_model"] = "gpt-3.5-turbo"

    # create a session state variable for message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # this is what displays the whole history but leaving this out just shares the most recent back and fourth
    #for message in st.session_state.messages:
    #    with st.chat_message(message["role"]):
    #        st.markdown(message["content"])

    if prompt := st.chat_input("Give me info about the hike"):
        # Clear previous messages
        st.session_state.messages.clear()


        # this is the append to have a larger context for user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # maybe i hardcode the response based on the outputs

        topics_mentioned = topics_mentioned_llm.invoke(prompt)
        st.session_state.topics_mentioned = topics_mentioned

    # require a start location 
    # require it's in california

        # Extracting and saving the variables
        location = topics_mentioned['location']
        #st.session_state.location = location

        travel_distance = topics_mentioned['travel_days_hours_distance']
        #st.session_state.travel_distance = travel_distance

        hiking_distance = topics_mentioned['hiking_distance_or_time']

        number_of_hikes = topics_mentioned['number_of_hikes'] 
        #st.session_state.number_of_hikes = number_of_hikes

        hike_difficulty = topics_mentioned['hiking_difficulty'] 

        # Print the variables to verify
        print("location:", location)
        print("travel_distance:", travel_distance)
        print("hiking_distance:", hiking_distance)
        print("number_of_hikes:", number_of_hikes)
        print("hike_difficulty:", hike_difficulty)

        to_be_displayed = []

        if location:
            start_location = start_location_tool_llm.invoke(prompt) 
            st.session_state.start_location = start_location
            #print(start_location)
            to_be_displayed.append(format_output(start_location, "Location"))
        
        if travel_distance:
            travel_dist = travel_dist_tool_llm.invoke(prompt)
            st.session_state.travel_dist = travel_dist
            #print(st.session_state.travel_dist)
            to_be_displayed.append(format_output(travel_dist, "Travel Distance"))

        if hiking_distance:
            hiking_dist = hiking_distance_tool_llm.invoke(prompt)
            st.session_state.hiking_dist = hiking_dist

            to_be_displayed.append(format_output(hiking_dist, "Hiking Distance"))

        if number_of_hikes:
            num_hikes = number_of_hikes_tool_llm.invoke(prompt)
            st.session_state.num_hikes = num_hikes
            #print(st.session_state.num_hikes )
            to_be_displayed.append(format_output(num_hikes, "Number of Hikes"))

        if hike_difficulty:
            difficulty = hike_difficulty_tool_llm.invoke(prompt)
            st.session_state.difficulty = difficulty
            to_be_displayed.append(format_output(difficulty, "Difficulty"))

        with st.chat_message("assistant"):
            st.markdown("Here's what i understood:")

        response = "\n\n".join(to_be_displayed)

        #with st.chat_message("assistant"):
        st.sidebar.markdown(response)

        with st.chat_message("assistant"):
            st.markdown("Are these all correct? I can make corrections or changes. (right now i don't take previous context into account so please create a new search)")

        # this is the append to have a larger context of response
        st.session_state.messages.append({"role": "assistant", "content": response})

     # Display the st.info box if the checkbox is checked
    if show_basic_info:
            st.sidebar.markdown("""
            ### Basic interactions
            This model takes your text and uses the following information to provide trail recommendations.
            You can use none, any or all of these criteria.
            - **Starting Location**: Where you want to start your trip from. you can name anywhere in California 
            - **Number of Hikes**: How many hikes you want to go on.
            - **Travel Distance**: How far you're willing to travel to find a suitable hike.
            - **Hiking Distance**: How long you want the hike to be.
            - **Hiking Difficulty**: The difficulty of the hike you're looking for.
            - **Top 5**: The top 5 closest trails based on the context you provided. so if you say ocean views or difficult mountain trail etc.  
            """)


        # this should reset the chat and response

        # change the chat bot so it returns the lsit of what we think was said 
        # and ask did i miss something?
        # anything else?
        # i can also take into account....
        # maybe we make a table of all the recommened hikes

if len(st.session_state.messages) >= 1: ##### I need to make this load after 

    print(st.session_state.messages)
    messages = st.session_state.messages
    # Extract user content
    query = next((message['content'] for message in messages if message['role'] == 'user'), None)

    topics_mentioned = st.session_state.topics_mentioned #topics_mentioned_llm.invoke(query)
    # require a start location 
    # require it's in california

    # Extracting and saving the variables
    if len(topics_mentioned) >= 1:
        location = topics_mentioned['location']
        travel_distance = topics_mentioned['travel_days_hours_distance']
        hiking_distance = topics_mentioned['hiking_distance_or_time']
        number_of_hikes = topics_mentioned['number_of_hikes'] 
        hike_difficulty = topics_mentioned['hiking_difficulty'] 


        # Initialize list to hold filtered DataFrames
        filtered_dfs = []

        # there's def a more effecient method where one by one we reduce the df based on the criteria

        # if else statments to run tools to get the data
        if location:
            start_location = st.session_state.start_location #start_location_tool_llm.invoke(query) 
            # past and plot the starting location
            # we need a starting location... maybe
            
            # add feature so if nothing then 50 miles is used
        if travel_distance:
            travel_dist = st.session_state.travel_dist #travel_dist_tool_llm.invoke(query)
            trails_travel, radius_degrees = filter_trails_for_travel_distance(st.session_state.trails, start_location, travel_dist['metric'], travel_dist['value'])
            filtered_dfs.append(trails_travel)

            # should add the the map a ring buffer that shows the set distance
            # Define the radius in degrees
            buffer_radius_meters = degrees_to_meters(start_location['Latitude'], start_location['Longitude'], radius_degrees)


        if travel_distance == False:
            trails_travel, radius_degrees = filter_trails_for_travel_distance(st.session_state.trails, start_location, "miles", 40)
            filtered_dfs.append(trails_travel)

            # Define the radius in degrees
            buffer_radius_meters = degrees_to_meters(start_location['Latitude'], start_location['Longitude'], radius_degrees)

            
        if hiking_distance:
            hiking_dist = st.session_state.hiking_dist #hiking_distance_tool_llm.invoke(query)
            trails_dist = filter_trails_for_hiking_distance(st.session_state.trails, hiking_dist['hike_metric'], hiking_dist['hike_distance'])
            filtered_dfs.append(trails_dist)
            
            # filter df based on hiking distance 

        if number_of_hikes:
            num_hikes = st.session_state.num_hikes #number_of_hikes_tool_llm.invoke(query)
            trails_num = filter_trails_for_num_hikes(st.session_state.trails, num_hikes['num_hikes'])
            filtered_dfs.append(trails_num)

        if hike_difficulty:
            difficulty = st.session_state.difficulty 
            trails_diff = filter_trails_for_difficulty(st.session_state.trails, difficulty['hike_difficulty'])
            filtered_dfs.append(trails_diff)

        #print('filtered_dfs list:', filtered_dfs)
        print('filtered_dfs list:', len(filtered_dfs))


        # Find intersection of all filtered DataFrames
        if filtered_dfs:
            trails_final = filtered_dfs[0]
            for df in filtered_dfs[1:]:
                trails_final = pd.merge(trails_final, df, how='inner')

            # Reset index if needed
            trails_final.reset_index(drop=True, inplace=True)

            if len(trails_final) == 0:
                st.markdown("too filtered")
        else:
            trails_final = []
            # on the main bit lets print the topics 
            st.markdown("no filters applied :(")

        

        st.markdown(f"Number of Trails found: **{len(trails_final)}**. Click on the trails below for more details")

        start_coords = [start_location['Latitude'], start_location['Longitude']]

        # Create a folium map centered around San Francisco
        m = folium.Map(width=800, location=start_coords, zoom_start=8)

        # Add a marker (dot) at the San Francisco coordinates
        folium.Marker(location=start_coords, popup='Start Location', tooltip='Start Location').add_to(m)  #### this doesn't seem to show up right now?

        # Add a buffer circle around the start location
        folium.Circle(
            location=start_coords,
            radius=buffer_radius_meters,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.05
        ).add_to(m)

        # Calculate the bounds of the circle and fit the map to those bounds
        circle_bounds = get_circle_bounds(start_coords, buffer_radius_meters)
        m.fit_bounds(circle_bounds)


        # let's add a number marker for prefered trails in order

        if len(trails_final) >= 1:

            collections = chroma_client.list_collections()
            if any(collection.name == 'trail_descriptions' for collection in collections):
                chroma_client.delete_collection('trail_descriptions')
                
            collection = chroma_client.create_collection(name="trail_descriptions") # need to learn how to create or replace or overwrite
            collection.add(
                documents=list(trails_final.description),
                ids=list(trails_final.index.astype(str))
            )
            top_matches = collection.query(
                query_texts=[query], # Chroma will embed this for you
                n_results=5 # how many results to return
            )
            
            ranking_df = pd.DataFrame({'id': top_matches['ids'][0], 'distance': top_matches['distances'][0]})

            #ranking_df = ranking_df[ranking_df['distance'] <= 1.36]

            ranking_df = ranking_df.drop_duplicates(subset=['id'], keep='last') #.unique() 
            ranking_df['id'] = ranking_df['id'].astype(int)

            # Add the 'rank' column with sequential numbers
            ranking_df['rank'] = range(1, len(ranking_df) + 1)

            # Merge trails_df with ranking_df to get the matching_df with rank column
            matching_df = trails_final.merge(ranking_df, left_index=True, right_on='id')
            #non_matching_df = trails_df[~trails_df.index.isin(ranking_df['id'])]

            

           # print("Matching df", matching_df)

            #matching_df = matching_df.sort_values(by='rank')


        # Add the first three records' geometries to the map with popups
            for _, row in trails_final.iterrows():
                # Define the popup content
                popup_content = f"""<h2>{row['name']}</h2>
                <br><strong>bikes allowed:</strong> {row['bicycle']}
                <br><strong>Trail length:</strong> {row['lengthmile']}
                <br><strong>Time to Complete (Hours):</strong> {round(row['completion_time_hours'])}
                <br><strong>Difficulty:</strong> {row['difficulty']}
                <br><strong>Description:</strong> {row['description']}"""
                # 	lengthmile	 completion_time_hours	difficulty
                popup = folium.Popup(popup_content, max_width=300)

                folium.GeoJson(
                    row.geometry,
                    name=row['name'],
                    tooltip=folium.Tooltip(row['name'] if row['name'] else 'No Name'),
                    style_function=outline_style_function
                ).add_to(m)

                # Add the green trail on top
                folium.GeoJson(
                    row.geometry,
                    name=row['name'],
                    tooltip=folium.Tooltip(row['name'] if row['name'] else 'No Name'),
                    style_function=trail_style_function,
                    highlight_function = highlight_function,
                    popup=popup
                ).add_to(m)
            
            if len(matching_df) >= 1:
                matching_df['start_location'] = matching_df['geometry'].apply(get_start_location)
                matching_df['start_location'] = matching_df['start_location'].apply(convert_to_list)
                matching_df['start_location'] = matching_df['start_location'].apply(invert_list_order)

                # Add markers with numbered green circle icons
                for index, row in matching_df.iterrows():
                    position = row['start_location'] #.values[0]
                    rank = int(row["rank"])
                    folium.Marker(
                        location= position, #loc['geometry'].apply(get_start_location).values[0],
                        icon=DivIcon(
                            icon_size=(30, 30),  # Adjust size as needed
                            icon_anchor=(15, 15),  # Center the icon
                            html=f'''
                            <div style="
                                background-color: green;
                                color: white;
                                border: 2px solid white;
                                border-radius: 50%;
                                width: 24px;
                                height: 24px;
                                text-align: center;
                                line-height: 24px;
                                font-weight: bold;
                                font-size: 12pt;">
                                {rank}
                            </div>
                            '''
                        ),
                            popup=popup
                    ).add_to(m)

        
        # Display the map
        folium_static(m, width=800)

    else:
    
        # Coordinates for San Francisco
        sf_coords = [37.7749, -122.4194]

        # Create a folium map centered around San Francisco
        m = folium.Map(width=800, location=sf_coords, zoom_start=8)

        # Add a marker (dot) at the San Francisco coordinates
        #folium.Marker(location=sf_coords, popup='San Francisco', tooltip='San Francisco').add_to(m)

        # Display the folium map using folium_static
        folium_static(m, width=800)
else:
    
    # Coordinates for San Francisco
    sf_coords = [37.7749, -122.4194]

    # Create a folium map centered around San Francisco
    m = folium.Map(width=800, location=sf_coords, zoom_start=8)

    # Add a marker (dot) at the San Francisco coordinates
    folium.Marker(location=sf_coords, popup='San Francisco', tooltip='San Francisco').add_to(m)

    # Display the folium map using folium_static
    folium_static(m, width=800)

# Center the map horizontally using Streamlit columns
#col1, col2, col3 = st.columns([1, 6, 1])
#with col2:
   # st_folium(m, width=900, height=500)
