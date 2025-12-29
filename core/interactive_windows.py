import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

def interactive_window(lista, parametro):
    root = tk.Tk()
    root.withdraw()
    seleccion = simpledialog.askstring(parametro, f"{parametro}:\n\n" + "\n".join(lista))
    root.destroy()
    if seleccion in lista:
        return seleccion
    return seleccion

def interactive_window_number(parametro):
    root = tk.Tk()
    root.withdraw()
    valor = simpledialog.askstring(parametro, parametro)
    root.destroy()
    return valor

def interactive_window_numbers(parametro):
    root = tk.Tk()
    root.withdraw()
    inicio = simpledialog.askstring(parametro, "Número inicial:")
    final = simpledialog.askstring(parametro, "Número final:")
    root.destroy()
    return inicio, final

def interactive_window_aviso(parametro):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Aviso", parametro)
    root.destroy()

def select_imaage(parametro):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.gif *.bmp *.tif *.tiff")],
        title=parametro
    )
    if file_path:
        print(f"Has seleccionado la siguiente imagen: {file_path}")
    else:
        print("No se seleccionó ninguna imagen.")
    return file_path

def select_path(parametro):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=parametro)
    if folder_path:
        print(f"You are select the following folder: {folder_path}")
    else:
        print("You did not select any folder.")
    return folder_path

def select_csv(parametro):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Archivos CSV", "*.csv")],
        title=parametro
    )
    if file_path:
        print(f"You are select the following CSV: {file_path}")
    else:
        print("You did not select any CSV.")
    return file_path
