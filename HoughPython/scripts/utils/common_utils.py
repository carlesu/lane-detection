import tkinter
import tkinter.filedialog
import pickle
import datetime
import os


def get_single_file(file_type=None):
    root = tkinter.Tk()
    if file_type != None:
        string_disp = 'Choose a ' + file_type + ' file'
        file_extension = file_type
        extension_str = "*." + file_extension
        file = tkinter.filedialog.askopenfilename(parent=root, title=string_disp,
                                                  filetypes=((file_type, extension_str), ("all files", "*.*")))
    else:
        string_disp = 'Choose a file'
        file = tkinter.filedialog.askopenfilename(parent=root, title=string_disp)
    root.withdraw()
    return file


def get_multiple_files():
    root = tkinter.Tk()
    files = tkinter.filedialog.askopenfilenames(parent=root, title='Choose several files',
                                                filetypes=(("all files", "*.*"), ))
    files_list = root.tk.splitlist(files)
    root.withdraw()
    return files_list


def get_single_file_with_extension(file_type=None, file_extension=None):
    root = tkinter.Tk()
    if file_type is not None:
        string_disp = 'Choose a ' + file_type + ' file'
    else:
        string_disp = 'Choose a file'
    if file_extension is not None:
        extension_str = "*." + file_extension
        file = tkinter.filedialog.askopenfilename(parent=root, title=string_disp,
                                                  filetypes=((file_type, extension_str), ("all files", "*.*")))
    else:
        file = tkinter.filedialog.askopenfilename(parent=root, title=string_disp)
    root.withdraw()
    return file


def get_multiple_files_with_extension(file_type=None, file_extension=None):
    root = tkinter.Tk()
    if file_type is not None:
        string_disp = 'Choose a ' + file_type + ' file'
    else:
        string_disp = 'Choose a file'
    if file_extension is not None:
        extension_str = "*." + file_extension
        file = tkinter.filedialog.askopenfilenames(parent=root, title=string_disp,
                                                   filetypes=((file_type, extension_str), ("all files", "*.*")))
    else:
        file = tkinter.filedialog.askopenfilenames(parent=root, title=string_disp)
    root.withdraw()
    return file


def get_single_folder():
    root = tkinter.Tk()
    folder = tkinter.filedialog.askdirectory(parent=root)
    root.withdraw()
    return folder


def file_save(file_extension=None):
    root = tkinter.Tk()
    if file_extension is not None:
        extension_str = "." + str(file_extension) + " Files"
        end_extension = "*." + str(file_extension)
        default_extension = [(extension_str, end_extension)]
        f = tkinter.filedialog.asksaveasfilename(parent=root, filetypes=default_extension,
                                                 defaultextension=end_extension)
    else:
        f = tkinter.filedialog.asksaveasfile(parent=root, mode='w')
    root.withdraw()
    return f


def save_pickle(data, save_path, file_title):
    timestr = datetime.datetime.now().strftime('%Y-%m-%d %Hh%Mm%S.%fs')
    if save_path[-1] != '/':
        save_path = save_path + '/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_name = save_path + file_title + '_' + timestr + '.pkl'
    pickle_file = open(file_name, "wb")
    pickle.dump(data, pickle_file)
    pickle_file.close()
    return file_name


def load_pickle(pkl_file):
    pickle_data = open(pkl_file, "rb")
    output = pickle.load(pickle_data)
    return output


def df_to_log(df):
    str_out = '\t' + df.to_string().replace('\n', '\n\t')
    return str_out


def true_copy(var_obj):
    out_var = pickle.loads(pickle.dumps(var_obj))
    return out_var