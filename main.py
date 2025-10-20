#!/usr/bin/env python3
"""
SmartImageSuite - Main Application Entry Point
A comprehensive image processing and analysis toolkit.

Author: [Your Name]
Date: [Current Date]
"""

import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def main():
    """
    Main function to initialize and run the SmartImageSuite application.
    """
    print("Welcome to SmartImageSuite!")
    print("A comprehensive image processing and analysis toolkit.")
    
    # Import and initialize the GUI interface
    try:
        from gui.interface import SmartImageSuiteGUI
        app = SmartImageSuiteGUI()
        app.run()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 



------------------------------
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:27:47 2022

@author: ajitesh.s
"""
import csv
import re
import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog


#For extracting all desired parameters from DGS file
class extract_param():
    def __init__(self,file):
        speclist=open('speclist.txt','r')

        outputfile_1 = open('temp.txt', 'w',newline='')
        for line in speclist:
            with open(file,"r",newline='') as csvfile:
                csv_reader = csv.reader(csvfile,dialect='excel',delimiter=',')
                for row in csv_reader:
                    if re.search(line.strip(), row[1].casefold()):
                        outputfile_1.writelines(row[1]+'\n')
            csvfile.close()
        outputfile_1.close()

        speclist.close()

        lines_seen = set() # holds lines already seen
        infile = open('temp.txt','r')
        outfile = open('param.txt','w')
        for line in infile:
            if line not in lines_seen: # not a duplicate
                if line.startswith('//') or line.startswith('FAIL'): # skip the comment lines
                    lines_seen.add(line)
                elif 'LOSS' in line: # skip the comment lines
                    lines_seen.add(line)
                elif 'Shielding' in line: # skip the comment lines
                    lines_seen.add(line)
                else:
                    outfile.write(line)
                    lines_seen.add(line)
        outfile.close()
        infile.close()
        
# Deviation calculation        
def Deviation(spec,file):
    line=0
    measure=0
    with open(file,"r",newline='') as csvfile:
        csv_reader = csv.reader(csvfile,dialect='excel',delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:     
            # print(spec,row[1])
            if row[1].casefold()+'\n'==spec.casefold():
                if float(row[2])>=float(row[3]) and float(row[2])<=float(row[4]):
                    line=line+1
                    mid = (float(row[3])+float(row[4]))/2
                    measure= measure+float(row[2])
    return mid-(measure/line)

# for seaching starting and end line nos. for any UN No. in loss file
def search_line(infile,UN):
    file = open(infile,'r')
    for count, line in enumerate(file):
        pass
    file.close()
    file = open(infile,'r')
    C_line=0
    start_line=0
    occur=0
    for row in file:
        if re.search(UN,row):
            C_line+=1
            occur+=1
            if occur==2:
                start_line=C_line
        
        elif start_line and C_line==int(count):
            end_line=C_line+1
            return start_line,end_line
        
        elif start_line and row.startswith('[UN'):
            end_line=C_line
            return start_line,end_line
        
        else:
           C_line+=1

# Main Class for GUI
class ALC:
    def __init__(self, master):
        master.title('Smart Loss Optimizer')
        master.resizable(False, False)
        photo = PhotoImage(file = 'icon.png')
        master.iconphoto(False, photo)
        self.window = Toplevel(master)
        self.window.title('Enter Your Password')
        self.window.lift(master)
        self.window.geometry('250x80')
        self.window.resizable(False, False)
        
        ttk.Label(self.window, text = 'Username:').grid(row = 0, column = 0, padx = 5, sticky = 'w')
        ttk.Label(self.window, text = 'Password:').grid(row = 1, column = 0, padx = 5, sticky = 'w')
        self.Username = ttk.Entry(self.window, width = 20, font = ('Arial', 11))
        self.Username.grid(row = 0, column = 1, padx = 5)
        self.Username.bind('<Return>', lambda e: self.authenticate())
        self.Password = ttk.Entry(self.window, width = 20, font = ('Arial', 11),show='*')
        self.Password.grid(row = 1, column = 1, padx = 5)
        self.Password.bind('<Return>', lambda e: self.authenticate())
        
        DGS_Frame=ttk.LabelFrame(master, height = 100, width = 200, text = ' DGS Logs ')
        DGS_Frame.grid(row = 0, column = 0, padx = 5)
        Loss_Frame=ttk.LabelFrame(master, height = 100, width = 200, text = ' Standard Loss File ')
        Loss_Frame.grid(row = 0, column = 1, padx = 5)
        DGS_Output=ttk.LabelFrame(master, height = 100, width = 200, text = ' Deviation List ')
        DGS_Output.grid(row = 1, column = 0, padx = 5)
        Loss_Output=ttk.LabelFrame(master, height = 100, width = 200, text = ' Adjusted Parameters List (0.2<=Deviation<=2.5) ')
        Loss_Output.grid(row = 1, column = 1, padx = 5)
        DGS_Scroll=Scrollbar(DGS_Output, orient='vertical')
        DGS_Scroll.grid(row=0,column=1, sticky='ns')
        Loss_Scroll=Scrollbar(Loss_Output, orient='vertical')
        Loss_Scroll.grid(row=0,column=1, sticky='ns')
                
        self.list_dgs=tk.Listbox(DGS_Frame, height=4, width = 53, selectmode='extended')
        self.list_dgs.grid(row = 0, column = 0, padx = 5, rowspan=2)
        ttk.Button(DGS_Frame, text = 'Select File', command = self.dgs_select).grid(row = 0, column = 1, padx = 5, pady = 5, sticky = 'e')
        ttk.Button(DGS_Frame, text = 'Clear', command = self.dgs_clear).grid(row = 1, column = 1, padx = 5, pady = 5, sticky = 'e')
        self.entry_loss = ttk.Entry(Loss_Frame, width = 39, font = ('Arial', 11))
        self.entry_loss.grid(row = 0, column = 0, padx = 5)
        ttk.Button(Loss_Frame, text = 'Browse File', command = self.loss_select).grid(row = 0, column = 2, padx = 5, pady = 5, sticky = 'e')
        self.combo_un = ttk.Combobox(Loss_Frame, state = 'readonly', width = 39, font = ('Arial', 10))
        self.combo_un.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = 'w')
        ttk.Label(Loss_Frame, text = 'Select UN:').grid(row = 1, column = 2, padx = 5, sticky = 'w')
        self.dgs_output = Text(DGS_Output, width = 55, height = 15, font = ('Arial', 10))
        self.dgs_output.grid(row = 0, column = 0, padx = 5, pady=5)
        DGS_Scroll.config(command=self.dgs_output.yview)
        self.dgs_output.config(yscrollcommand = DGS_Scroll.set)
        self.loss_output = Text(Loss_Output, width = 55, height = 15, font = ('Arial', 10))
        self.loss_output.grid(row = 0, column = 0, padx = 5, pady=5)
        Loss_Scroll.config(command=self.loss_output.yview)
        self.loss_output.config(yscrollcommand = Loss_Scroll.set)
        self.loss_output.tag_config("blue", foreground="blue")
        ttk.Button(master, text = 'Find Deviation', command = self.Deviation_Button).grid(row = 2, column = 0, padx = 5, pady = 5, sticky = 'nsew')
        ttk.Button(master, text = 'Adjust Deviation', command = self.Adjust).grid(row = 2, column = 1, padx = 5, pady = 5, sticky = 'nsew')
        ttk.Label(master, text = 'Created by : Ajitesh Kumar Sagar (ajitesh.s@samsung.com) - SIEL MX-R&D', foreground= '#0000FF', font = ('Arial', 9, 'italic')).grid(row = 3, column = 0, columnspan=2, padx = 5, sticky = 'se')
        ttk.Button(self.window, text = 'Submit', command = self.authenticate).grid(row = 3, column = 0, columnspan=2, padx = 5, pady = 5)
        self.window.protocol("WM_DELETE_WINDOW", self.disable_event)
        self.window.grab_set()
    
    def disable_event(self):
        pass


    def authenticate(self):
        user=self.Username.get()
        password=self.Password.get()
        if user=='admin' and password =='1111':
            self.window.grab_release()
            self.window.destroy()
        else:
            messagebox.showinfo(title = 'Error', message = 'Wrong Username or Password!')
                
    def dgs_select(self):
        self.list_dgs.insert(0,filedialog.askopenfilename(initialdir="E:\Projects\Auto Loss Correction"))
            
    def dgs_clear(self):
        self.list_dgs.delete(0,END)
        self.dgs_output.delete(1.0, END)
            
    def loss_select(self):
        self.entry_loss.delete(0, END)
        self.entry_loss.insert(0,filedialog.askopenfilename(initialdir="E:\Projects\Auto Loss Correction"))
        LossFile=self.entry_loss.get()
        UN_list=[]
        with open(LossFile,"r",newline='') as csvfile:
            csv_reader = csv.reader(csvfile,dialect='excel',delimiter='=')
            for row in csv_reader:
                if row[0]=='COUNT':
                    UN_Count=int(row[1])
                    for i in range(UN_Count):
                        row=next(csv_reader)
                        UN_list.append(row[1])
        self.combo_un['values']=UN_list
                
                
    def Deviation_Button(self):
        self.dgs_output.delete(1.0, 'end')
        file_list=self.list_dgs.get(0,END)
        x=len(file_list)
        with open('Temp1.csv','w') as file:
            try:
                for i in range(x):
                    csvfile=open(file_list[i],'r')
                    for row in csvfile:
                        file.write(f',{row}')
                    csvfile.close()
                file.close()
            except Exception as e:
                self.dgs_output.insert(1.0, e)
        
        extract_param('Temp1.csv')
        self.dgs_output.insert(1.0, 'Deviation list as below:\n\n')
        param_file=open('param.txt','r')
        for line in param_file:
            x=Deviation(line,'Temp1.csv')
            self.dgs_output.insert(END,'Test Item : {}Deviation : {:.3f} \n\n'.format(line,x))
        param_file.close()
        
    def Adjust(self):
        self.loss_output.delete(1.0, 'end')
        UN=self.combo_un.get()
        LossFile=self.entry_loss.get()
        warning_list=[]
        start,end=search_line(LossFile,UN) #To Search lines to edit in loss file
        self.loss_output.insert(END, 'Below parameters updated:\n')
        self.loss_output.insert(END,f'Start Line : {start}\nEnd Line : {end}\n\n')
        self.loss_output.insert(END,'Parameter\t\t\t\tValue\n\n')
        
        paramfile=open('param.txt','r')
        for line in paramfile:
            x=Deviation(line,'Temp1.csv')
            if abs(x)>2.5:
                warning_list.append(line.rstrip('\n'))
                warning_list.append(round(float(x),3))
                
            elif abs(x)>=0.2 and abs(x)<=2.5:
                count = 0
                with open('Analogy.txt', 'r',newline='') as analogyfile:
                    analogy=csv.reader(analogyfile,delimiter='=')
                    for key in analogy:
                        if re.search(key[0],line.casefold()):
                            str1=key[1]
                            with open('RF Bands.txt','r') as bandfile:
                                csv_reader = csv.reader(bandfile,dialect='excel',delimiter='=')
                                for row in csv_reader:
                                    if row[0] in line:
                                        loss_param=row[1]+str1
                                        with open(LossFile,'r') as f:
                                            csv_reader = csv.reader(f,dialect='excel',delimiter='=')
                                            with open('data.txt', 'w',newline='') as f1:
                                                writer=csv.writer(f1,delimiter='=')
                                                for row in csv_reader:
                                                    count += 1
                                                    if count>start and count<end: 
                                                        if row[0] == loss_param:
                                                            row=[row[0],round(((float(row[1]))+x),3)]
                                                            self.loss_output.insert(END,f'{row[0]}\t:\t{row[1]}\n')
                                                            # print(row)
                                                    writer.writerow(row)
                                            f1.close()
                                        f.close()
                                        os.remove(LossFile)
                                        os.rename('data.txt', LossFile)
        bandfile.close()
        analogyfile.close()
        paramfile.close()
        if len(warning_list):
            self.loss_output.insert(END,'\nWarning : Below parameters deviation is above 2.5\n', 'blue')        
            for values in warning_list:
                self.loss_output.insert(END,f'\n {values}\n', 'blue')        
        messagebox.showinfo(title = 'Loss File Updated', message = 'Standard Loss File updated successfully!')
        os.remove('temp.txt')
        os.remove('Temp1.csv')
        os.remove('param.txt')
        
def main():            
    root = Tk()
    ALC(root)
    root.mainloop()
    
        
if __name__ == "__main__": main()
