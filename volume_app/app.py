from shiny import App, render, ui, reactive
#import matplotlib.pyplot as plt
#import numpy as np
from SimuInf.confset import confset
from SimuInf.plotting import confset_plot
import joblib
from nilearn.image import get_data
from pathlib import Path

# page_fluid creates a blank page while ui.page_sidebar creates the sidebar page 
app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("threshold", "Activation Threshold",  min=0,
                            max=20,value=2,step=0.5,width="100%"),
            ui.input_slider("cut1", "Cut Coordinate of Slice 1",  min=0,
                            max=90,value=15,step=1,width="100%"),
            ui.input_slider("cut2", "Cut Coordinate of Slice 2",  min=0,
                            max=90,value=40,step=1,width="100%"),
            ui.input_slider("cut3", "Cut Coordinate of Slice 3",  min=0,
                            max=90,value=50,step=1,width="100%"),   
            ui.input_slider("cut4", "Cut Coordinate of Slice 4",  min=0,
                            max=90,value=60,step=1,width="100%"),
            ui.input_action_button('reset','Reset!')                           
        ),
        # reduce height to remove the extra white space
        ui.output_plot('plot_z', height='225px'),
        ui.output_plot('plot_x', height='225px'),
        ui.output_plot('plot_y', height='225px'),
        gap='0px'
        ),
        title='Find Activated Brain Regions')


# need to put these inside app folder if you want to host it
est = joblib.load(Path(__file__).parent / "hcp/est")
masker = joblib.load(Path(__file__).parent /"hcp/masker")
lower = joblib.load(Path(__file__).parent /"hcp/lower")
upper = joblib.load(Path(__file__).parent /"hcp/upper")
brain = joblib.load(Path(__file__).parent /"hcp/brain")


def server(input, output, session):
    @reactive.calc
    def get_set_unmasked():
        set_masked = list(confset(est, lower, upper, threshold=input.threshold()))
        set_unmasked =  [get_data(masker.inverse_transform(set)) for set in set_masked] 
        return set_unmasked
    
    @reactive.calc
    def get_cuts_ls():
        set_masked = list(confset(est, lower, upper, threshold=input.threshold()))
        set_unmasked =  [get_data(masker.inverse_transform(set)) for set in set_masked] 
        return [input.cut1(), input.cut2(), input.cut3(), input.cut4()]
    
    @render.text
    def txt1():
        set_unmasked = get_set_unmasked()
        return f"print {len([set_unmasked])}"

    @render.plot(alt="A plot")
    def plot_z():
        set_unmasked = get_set_unmasked()
        # tight layout and figure size did not help the spacing in the element
        confset_plot([set_unmasked], [''] , nrow=1, figsize=(8,2), fontsize=15, background =brain, cuts = get_cuts_ls(), ticks=False) 
        #plt.tight_layout()
        
    @render.plot(alt="A plot")
    def plot_x():
        set_unmasked = get_set_unmasked()
        confset_plot([set_unmasked], [''] , nrow=1, figsize=(8,2), fontsize=15, background =brain, display_mode='x', cuts = get_cuts_ls(), ticks=False)   
        #plt.tight_layout()
    @render.plot(alt="A plot")
    def plot_y():
        set_unmasked = get_set_unmasked()
        confset_plot([set_unmasked], [''] , nrow=1, figsize=(8,2), fontsize=15, background =brain, display_mode='y', cuts = get_cuts_ls(), ticks=False)  
        #plt.tight_layout()
    @reactive.Effect
    @reactive.event(input.reset)
    def on_reset_click():
        ui.update_slider('threshold', value=2)
        ui.update_slider('cut1', value=15)
        ui.update_slider('cut2', value=40)
        ui.update_slider('cut3', value=50)
        ui.update_slider('cut4', value=60)

    


app = App(app_ui, server)








