from mesa.visualization.ModularVisualization import ModularServer

from agent import Obstacle
from berrymodel import BerryCollectionModel
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.UserParam import Slider
from agent import PreyAgent, BerryAgent

NUMBER_OF_CELLS = 4

SIZE_OF_CANVAS_IN_PIXELS_X = 500
SIZE_OF_CANVAS_IN_PIXELS_Y = 500

simulation_params = {
    "number_of_prey": Slider("number_of_prey", 1, 0, 10, 1),
    "number_of_berries": Slider("number_of_berries", 1, 0, 10, 1),
    #"number_of_predator": {
    #    "type": "SliderInt",
    #    "value": 10,
    #    "label": "initial number of predators",
    #    "min": 10,
    #    "max": 200,
    #    "step": 1,
    #    },
    "width": NUMBER_OF_CELLS,
    "height": NUMBER_OF_CELLS,
}


def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, PreyAgent):
        portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5, "Color": "green", "Layer": 1}
        
    if isinstance(agent, BerryAgent):
        portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5, "Color": "blue", "Layer": 1}
    
    if isinstance(agent, Obstacle):
        portrayal = agent.draw(agent)
            

    return portrayal

grid = CanvasGrid(
    agent_portrayal,
    NUMBER_OF_CELLS,
    NUMBER_OF_CELLS,
    SIZE_OF_CANVAS_IN_PIXELS_X,
    SIZE_OF_CANVAS_IN_PIXELS_Y,
)

server = ModularServer(
    BerryCollectionModel, [grid], "Berry Collection Model", simulation_params,
)
server.port = 8889  # The default
#server.launch()