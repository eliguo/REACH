import os
import zipfile
import requests
from tqdm import tqdm

models_dir = os.path.join(os.path.dirname(__file__), 'models')
zip_file_path = os.path.join(models_dir, 'models.zip')
predictor_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
facenet_model_path = os.path.join(models_dir, 'facenet_keras.h5')

def download_file_with_progress(url, filename):
    """
    Downloads a file with a progress meter.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR, something went wrong")
        

def download_models(overwrite=False):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        

        if not os.path.exists(zip_file_path) or overwrite:
            print("Downloading models...")
            url = 'https://pitt-my.sharepoint.com/:u:/g/personal/sah273_pitt_edu/EQst3JgthotLrFlbQgy9HT0B07wcaQxLm_Jxrv78JC-vRg?download=1'
            download_file_with_progress(url, zip_file_path)

        # Extract models from the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(__file__))
        os.remove(zip_file_path)



def PyAFAR_GUIRUN():
    # -*- coding: utf-8 -*-
    """
    Created on Sun Apr  2 21:05:38 2023

    @author: ntweat
    """

    import FreeSimpleGUI as sg
    import time
    #import pdb
    import datetime
    import os
    import threading
    import requests
    #import singlecode
    import pandas as pd
    #import multiprocessing
    import sys
    import signal


    use_custom_titlebar = True
    infant_occ_aus = [1,2, 3,4, 6, 9, 12, 20, 28]
    infant_int_aus = []
    adult_occ_aus = [1,2,4,6,7,10,12,14,15,17,23,24]
    adult_int_aus = [6,10,12,14,17]

    total_occ = list(set(infant_occ_aus) | set(adult_occ_aus))
    total_int = list(set(infant_int_aus) | set(adult_int_aus))
    models = ['Infant/au_1.txt', 'Infant/au_2.txt','Infant/au_3.txt',
            'Infant/au_4.txt','Infant/au_6.txt','Infant/au_9.txt',
            'Infant/au_12.txt','Infant/au_20.txt','Infant/au_28.txt','facenet_keras.h5', 
            'shape_predictor_5_face_landmarks.dat', 'adult/occ/au_1.h5', 
            'adult/occ/au_2.h5','adult/occ/au_4.h5','adult/occ/au_6.h5',
            'adult/occ/au_7.h5','adult/occ/au_10.h5','adult/occ/au_12.h5',
            'adult/occ/au_14.h5','adult/occ/au_15.h5','adult/occ/au_17.h5',
            'adult/occ/au_23.h5','adult/occ/au_24.h5']
    model_path = os.path.join(os.path.dirname(__file__), 'models')

    ps_script = "downloadModelss.ps1"

    class StoppableThread(threading.Thread):
        def __init__(self, target=None, args=(), kwargs={}):
            super().__init__(target=target, args=args, kwargs=kwargs)
            self._stop_event = threading.Event()

        def stop(self):
            self._stop_event.set()

        def kill(self):
            # Schedule a function call that stops the thread after 1 second
            threading.Timer(1, self._stop).start()

        def run(self):
            while not self._stop_event.is_set():
                if self._target:
                    self._target(*self._args, **self._kwargs)
                else:
                    self._bootstrap()
                    
    class FileProcessingThread(threading.Thread):
        def __init__(self, input_file, AUs, GPU, batchsize, output_file):
            threading.Thread.__init__(self)
            self.input_file = input_file
            self.AUs = AUs
            self.GPU = GPU
            self.batchsize = batchsize
            self.output_file = output_file
            self.running = True
            self.stop_event = threading.Event()
        
        def run(self):
            try:
                cv = singlecode.run_AFAR(self.input_file, self.AUs, self.GPU, self.batchsize)
                df = pd.DataFrame.from_dict(cv)
                df.to_csv(self.output_file)
            except KeyboardInterrupt:
                self.running = False
                return
        
        def terminate(self):
            self.running = False
            
        def stop(self):
            self.stop_event.set()



                    
    def set_default_values(window):
        window["-GPU-"].update(True)
        #window["-GPU IDS-"].update(-1)
        #window["-generate_dynamics-"].update(True)
        window["-generate_landmarks-"].update(True)
        #window["-save_normalized_videos-"].update(True)
        window["-batchsize-"].update(15)
        for au in adult_occ_aus:
            window["-occ_{}-".format(au)].update(True)
        #for au in int_aus:
        #   window["-int_{}-".format(au)].update(True)
        return window
        
    '''    
    def au_occurrence_flags(occ_aus):
        output = list()
        for au in occ_aus:
            output.append(sg.Checkbox('AU '+str(au), default=True, k='-occ_{}-'.format(au)))
        return output
    '''
    def au_occurrence_flags(occ_aus):
        col1 = [[sg.Checkbox('AU ' + str(au), default=True, k='-occ_{}-'.format(au))] for au in occ_aus[:len(occ_aus)//3]]
        col2 = [[sg.Checkbox('AU ' + str(au), default=True, k='-occ_{}-'.format(au))] for au in occ_aus[len(occ_aus)//3:2*len(occ_aus)//3]]
        col3 = [[sg.Checkbox('AU ' + str(au), default=True, k='-occ_{}-'.format(au))] for au in occ_aus[2*len(occ_aus)//3:]]
        
        output = [sg.Column(col1), sg.Column(col2), sg.Column(col3)]
        return output


    def au_intensity_flags(int_aus):
        output = list()
        for au in int_aus:
            output.append(sg.Checkbox('AU '+str(au), default=True, k='-int_{}-'.format(au)))
        return output


    def about_pyafar():
        current_year = datetime.datetime.now().year
        message = '''PyAFAR is a Python-based, open-source facial action unit detection library for use with adults and infants.
        
        PyAFAR is developed by Affect Analysis Group at University of Pittsburgh.
    For more information, visit https://pyafar.org/.

    Copyright © {year} University of Pittsburgh.

    This software is licensed under the MIT License. 
    '''.format(year=current_year)
        sg.popup(message, title='About PyAFAR' )



    NAME_SIZE = 35
    def process_file(input_file, AUs, GPU, batchsize, max_frames, output_file):
        #if not GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        #else:
        #    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        #run_AFAR(filename, AUs, GPU, batch_size, max_frames, AU_Ints=[]):
        import PyAFAR_GUI.infant_afar as sn
        try:
            cv = sn.infant_afar(input_file, AUs, GPU, max_frames)
            df = pd.DataFrame.from_dict(cv)
            df.to_csv(output_file)
        except KeyboardInterrupt:
            return
        
    def process_adult(input_file, AUs, GPU, batchsize, max_frames,  PID, AU_Int, output_file):
        if not GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        #run_PyAFAR(filename, AUs, GPU, max_frames, AU_Int, batch_size):
        import PyAFAR_GUI.adult_afar as sn
        try:
            cv = sn.adult_afar(input_file, AUs, GPU, max_frames, AU_Int, batchsize, PID)
            df = pd.DataFrame.from_dict(cv)
            df.to_csv(output_file)
        except KeyboardInterrupt:
            return

    def set_false(window):
        for au in total_occ:
            window['-occ_{}-'.format(au)].update(value=False, visible=False)
        for au in total_int:
            window['-int_{}-'.format(au)].update(value=False, visible=False)
        return window
        
    def set_adult(window):
        window = set_false(window)
        window['-Mode-'].update(value="Adult Mode")
        window['-AdvInt-'].update(visible=True)
        for au in adult_occ_aus:
            window['-occ_{}-'.format(au)].update(value=True, visible=True)
        for au in adult_int_aus:
            window['-int_{}-'.format(au)].update(value=True, visible=True)
        return window

    def set_infant(window):
        window = set_false(window)
        window['-Mode-'].update(value="Infant Mode")
        window['-AdvInt-'].update(visible=False)
        for au in infant_occ_aus:
            window['-occ_{}-'.format(au)].update(value=True, visible=True)

        return window

    def make_window(theme=None):

        NAME_SIZE = 30


        def name(name, padding_space=None, Keyd = None, tool=None):
            if padding_space is None:
                dots = NAME_SIZE-len(name)-2
            else:
                dots = padding_space
            #dots = 0
            if Keyd is None and tool is None:
                return sg.Text(name + ' ' + ' '*dots, size=(NAME_SIZE,1), justification='l',pad=(0,0), font='Courier 10')
            elif Keyd is None:
                return sg.Text(name + ' \u24d8 ' + ' '*dots, size=(NAME_SIZE,1), justification='l',pad=(0,0), font='Courier 10', tooltip=tool)
            else:
                return sg.Text(name + ' ' + ' '*dots, size=(NAME_SIZE,1), justification='l',pad=(0,0), font='Courier 10', key=Keyd)


        sg.theme(theme)

        # NOTE that we're using our own LOCAL Menu element
        if use_custom_titlebar:
            Menu = sg.MenubarCustom
        else:
            Menu = sg.Menu

        treedata = sg.TreeData()
        basic_layout = [
                        [name("Input Videos"), sg.Input(key='-VIDEO INPUT-'), sg.Button("Browse", key="video_browser")],
                        [name('Save Folder'), sg.Input(key='-SAVE DIR-'), sg.Button("Browse", key="save_browser")],
                        [sg.Checkbox('Overwrite \u24d8', default=False, k='-Overwrite-', tooltip="Overwrite Existing CSV files.")],
                        [name("PyAFAR version", tool="Adult or Infant Models to us (click radio button till Selected mode changes)"), sg.Radio('Adult', "RADIO1",enable_events=True, default=True, key="Adult"), sg.Radio('Infants', "RADIO1",enable_events=True, default=False, key="Infant")],
                        [name("Selected Mode", tool="Selected models"), sg.Text("Adult Mode", text_color='red',  key='-Mode-', size=(50,1), font=('Helvetica', 12, 'bold'))],
                        [sg.Button('Process'), sg.Button('Exit')]]
        
        
        

        

                
        #'''

        advance_layout = [
                    
                        [sg.Checkbox('Use GPU  \u24d8', default=True, k='-GPU-',tooltip='Will use default GUP 0, if available', disabled=False)],
                        [sg.Text("Batch size (prediction) \u24d8", tooltip="Number of images per batch for prediction. Higher number will process faster, but need more resources. (this is dependent on GPU)"),  sg.Spin(values=[i for i in range(5001)], size=(4,1), k='-batchsize-', initial_value=100)],
                        [sg.Text("Preprocessing Batch size  \u24d8", tooltip="Number of images for preprocessing before prediction. (this is dependent on RAM)"),  sg.Spin(values=[i for i in range(5001)], size=(4,1), k='-maxframes-', initial_value=1000)],
                        #[sg.Checkbox('Head & Face dynamics \u24d8', default=True, k='-generate_dynamics-', tooltip="Head pose.")],
                        #[sg.Checkbox('Facial Landmarks', default=True, k='-generate_landmarks-')],
                        [sg.Checkbox('Person Tracking (only for Adults) \u24d8', default=False, k='-PID-', tooltip="Identify people in video (disabling this will speed up the processing)", disabled=True)],
                        #[sg.Checkbox('Save aligned videos', default=True, k='-save_normalized_videos-')],
                        [name('AU occurrence'), *au_occurrence_flags(total_occ)],
                        [name('AU intensity', Keyd='-AdvInt-'), *au_intensity_flags(total_int)],
                        [sg.Button('Process', key="Process_adv"), sg.Button('Exit', key="Exit_adv")]]
        

        layout = [[Menu([['File', ['Exit']], ['About', ['About PyAFAR' ]]],  k='-CUST MENUBAR-',p=0)],
                [sg.T('PyAFAR', font='_ 14', justification='c', expand_x=True)]]
                
        layout +=[[sg.TabGroup([[  sg.Tab('Basic settings', basic_layout),
                                sg.Tab('Advance settings', advance_layout)]], key='-TAB GROUP-', expand_x=True, expand_y=True)
                ]]

        window = sg.Window('PyAFAR v0.1', layout, finalize=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, keep_on_top=False, use_custom_titlebar=use_custom_titlebar,resizable=False)


        return window
        
    def initiate_progress():
        layout = [[sg.Text("Processing files...", key="-STATUS-")], 
            [sg.Text("Completed: 0%", key="-PERCENTAGE-")],
        [sg.ProgressBar(100, orientation='h', s=(20,20), k='-PBAR-')],
        [sg.Cancel()]]
        progress_window = sg.Window("Running PyAFAR", layout, finalize=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, keep_on_top=False,resizable=False)
        return progress_window


    window = make_window()
    window = set_adult(window)
    sg.set_options(use_custom_titlebar=use_custom_titlebar)


    while True:
        
        mis = False
        event, values = window.read()
        #print(values['RADIO1'])
        print(event)
        if event == sg.WIN_CLOSED or event == 'Exit' or event == 'Exit_adv':
            window.close()
            break
        elif event == 'About PyAFAR':
            about_pyafar()
        elif event == 'Adult' or event == 'Infant':
            print(event)
            if values['Adult']:
                window=set_adult(window)
            else:
                window=set_infant(window)

            event, values = window.read()

        #elif event == 'Version':
        #    sg.popup(__file__, sg.get_versions(), keep_on_top=False, non_blocking=True)
        elif event in ["video_browser", "save_browser"]:
            # print("[LOG] Clicked Open Folder!")
            folder_or_file = sg.popup_get_folder('Choose your folder', no_window=False, keep_on_top=False)
            #sg.popup("You chose: " + str(folder_or_file), keep_on_top=True)
            if event == "video_browser":
                window['-VIDEO INPUT-'].update(folder_or_file)
            elif event == "save_browser":
                window['-SAVE DIR-'].update(folder_or_file)

            # print("[LOG] User chose folder: " + str(folder_or_file))
        elif event == "Restore Defaults":
            # pdb.set_trace()
            window = set_default_values(window)
            event, values = window.read()
            
        elif event == 'Process' or event == 'Process_adv':
            input_dir = values['-VIDEO INPUT-'].strip()
            output_dir = values['-SAVE DIR-'].strip()
            message =[]
            
            if not input_dir or not output_dir:
                mis=True
                message.append(f"Input and output directories cannot be blank")
            elif not os.path.exists(input_dir) or not os.path.exists(output_dir):
                mis=True
                message.append(f"Input and output directories must exist")
            elif input_dir == output_dir:
                mis=True
                message.append(f"Input and output directories must be different")
                
                

            #'''
            print(model_path)
            if not os.path.exists(model_path):
                mis = True
                message.append(f'The models folder does not exist. Please download the models using the included script.')
            else:
                missing_files = [file for file in models if not os.path.exists(os.path.join(model_path, file))]
                
                if missing_files:
                    mis = True
                    missing_files_str = '\n'.join(missing_files).replace(",", "\n")  # Replace commas with new lines
                    message.append(f'The following models are missing, please download them using the included script:\n{missing_files_str}')
                #'''
            if mis:
                msg = "Following Errors were encountered: \n\n"
                for i in range(len(message)):
                    msg = msg + str(i+1) +". "+message[i] +"\n\n"
                    
                    
                sg.popup(msg, title="Error Occurred", icon='error')
            else:
                progress_window = initiate_progress()
                AUs = []
                AU_Int = []
                for au in total_occ: 
                    if window['-occ_'+str(au)+'-'].get(): 
                        AUs.append("au_"+str(au))

                for au in total_int:
                    if window['-int_'+str(au)+'-'].get():
                        AU_Int.append("au_"+str(au)) 

                print(AUs)
                num_files = len(os.listdir(input_dir))
                for i, file in enumerate(os.listdir(input_dir)):
                    per_done = i*100/len(os.listdir(input_dir))
                    progress_event, progress_values = progress_window.read(timeout=10)
                    if progress_event == 'Cancel'  or progress_event == sg.WIN_CLOSED:
                        progress_window.close()
                        break
                    progress_window['-PBAR-'].update(int(per_done))
                    progress_window["-PERCENTAGE-"].update(f"Completed: {i*100/len(os.listdir(input_dir)):.2f}%")
                    progress_window["-STATUS-"].update(f"Processing {file}...")
                    if not values["-Overwrite-"] and os.path.exists(os.path.join(output_dir, os.path.splitext(file)[0]+".csv")):
                        progress_window["-STATUS-"].update(f"Skipping {file}...")
                        progress_event, progress_values = progress_window.read(timeout=100)
                        continue
                        
                    
                    #time.sleep(1)
                    #try:
                    '''
                    cv = singlecode.run_AFAR(os.path.join(input_dir, file), AUs, values["-GPU-"], values["-batchsize-"] )
                    df = pd.DataFrame.from_dict(cv)
                    df.to_csv(os.path.join(output_dir, os.path.splitext(file)[0]+".csv"))
                    '''
                    if not values['Adult']:
                        processing = threading.Thread(target=process_file, args=(os.path.join(input_dir, file), AUs, values["-GPU-"], values["-batchsize-"], values['-maxframes-'],os.path.join(output_dir, os.path.splitext(file)[0]+".csv")))
                    else:
                        processing = threading.Thread(target=process_adult, args=(os.path.join(input_dir, file), AUs, values["-GPU-"], values["-batchsize-"], values['-maxframes-'],  values["-PID-"],AU_Int, os.path.join(output_dir, os.path.splitext(file)[0]+".csv")))
                    #processing = FileProcessingThread(
                    #eventer = threading.Event()
                    #processing = multiprocessing.Process(target=process_file, args=(os.path.join(input_dir, file), AUs, values["-GPU-"], values["-batchsize-"], os.path.join(output_dir, os.path.splitext(file)[0]+".csv")))
                    #processing = StoppableThread(target=process_file, args=(os.path.join(input_dir, file), AUs, values["-GPU-"], values["-batchsize-"], os.path.join(output_dir, os.path.splitext(file)[0]+".csv")))
                    processing.setDaemon(True)
                    processing.start()
                    while processing.is_alive():
                        progress_event, progress_values = progress_window.read(timeout=100)
                        if progress_event == 'Cancel'  or progress_event == sg.WIN_CLOSED:
                            progress_window["-STATUS-"].update(f"Stopping the process please wait")
                            progress_event, progress_values = progress_window.read(timeout=1)
                            
                            thread_pid = os.getpid()
                            print(thread_pid)
                            
                            time.sleep(5)
                            os.kill(thread_pid, signal.SIGTERM)
                            #sys.exit(1)
                            
                            progress_window.close()
                            
                            break
                    #processing.join()
                        
                    
                progress_window["-STATUS-"].update(f"Directory Processing Completed!!")
                progress_window["-PERCENTAGE-"].update(f"Completed: 100.00%")
                event, values = progress_window.read()
                if event == sg.WIN_CLOSED: 
                    progress_window.close()
                progress_window.close()

    window.close()


