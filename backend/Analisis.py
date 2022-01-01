import reportes.TendenciaPais1 as case1

def redirigirAnalisis(filepath,caso,name,param):
    if caso == 1:
        #Tendencia de la infección por Covid-19 en un País.
        x_celda = param['tiempo']
        y_celda = param['confirmados']
        celda_pais = param['celda_pais']
        nombre_pais = param['nombre_pais']
        lista = case1.analizar(filepath,x_celda,y_celda,celda_pais,nombre_pais)
        return lista

    return {}