import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MainService } from '../servicios/main.service';

@Component({
  selector: 'app-parametros',
  templateUrl: './parametros.component.html',
  styleUrls: ['./parametros.component.css']
})
export class ParametrosComponent implements OnInit {

  columnas:Array<any> = [];
  nombreArchivo:string = '';
  listaAnalisis:any;
  listaNombres:any;
  haydatos:boolean = false;
  listaParametros:Array<any> = [];
  listaParametros_num:Array<any> = [];
  listaParametros_text:Array<any> = [];
  listaParametros_opc:Array<any> = [];
  casoActual:any;
  isOkey:boolean = true;
  celdas_faltantes:string = '';
  hayError:string = '';

  constructor(private conexion:MainService,private router:Router) { }

  ngOnInit(): void {
    let data = this.conexion.obj_carga_to_parameter.data
    if (this.columnas.length == 0) { 
      // console.log('no hay data')
      this.haydatos = false
    }else{
      this.columnas = data.columnas;
      this.haydatos = true
    }
    this.hayError ='';
    this.conexion.getDataParameter()
    .subscribe(data =>{
      // console.log(data)
      if(data.status == 200){
        this.nombreArchivo = data.body.fileName
        if (this.columnas.length == 0){
          this.columnas = data.body.columnas;
          if(this.columnas.length == 0){
            this.haydatos = false;
          }else{
            this.haydatos = true;
          }  
        }
        this.listaAnalisis =  data.body.listaAnalisis
        this.listaNombres = data.body.listaNombres
      }else{
          let estado = data.status
          let mensaje =  data.body.error
          this.hayError = `Error ${estado} : ${mensaje}`
      }
    },
    err => {
      console.log(err)
      this.hayError = `Error ${err.status} : ${err.message}`
    }
    )
  }

  cambioAnalisis(evento:any){
    this.casoActual = this.listaAnalisis[evento.target.value]
    this.listaParametros = this.casoActual['parametros']
    this.listaParametros_num = this.revisarArray(this.casoActual['parametros_numericos'])
    this.listaParametros_text = this.revisarArray(this.casoActual['parametros_texto'])
    this.listaParametros_opc = this.revisarArray(this.casoActual['opcionales'])
  }

  revisarArray(arr:any){
    if(arr == undefined){
      return []
    }else{
      return arr
    }
  }

  subirParametros(form:any){
    this.celdas_faltantes = ''
    this.isOkey = true;
    this.hayError ='';

    let parametros = {}
    this.listaParametros.forEach((paramName,index,array)=>{
      //@ts-ignore
      parametros[paramName] = this.checkOpcionales(paramName,form.value[paramName])
    })
    this.listaParametros_num.forEach((paramName,index,array)=>{
      //@ts-ignore
      parametros[paramName] = this.checkOpcionales(paramName,form.value[paramName])
    })
    this.listaParametros_text.forEach((paramName,index,array)=>{
      //@ts-ignore
      parametros[paramName] = this.checkOpcionales(paramName,form.value[paramName])
    })
    // console.log('params',parametros)
    let todoJson = {
      'caso': this.casoActual.caso,
      'name': this.casoActual.name,
      'parametros': parametros
    }
    if (this.isOkey) {
      //se puede subir
      console.log('todo esta fino')
      this.conexion.sendParameters(todoJson)
      .subscribe(data =>{
        console.log('data recived', data)
        if (data.status == 200) {
          this.router.navigate(['/reporte'])
        }else{
          let estado = data.status
          let mensaje =  data.body.error
          this.hayError = `Error ${estado} : ${mensaje}`
        }
      },
      err => {
        console.log(err)
        this.hayError = `Error ${err.status} : ${err.message}`
      }
      )
    }
  }

  checkOpcionales(name:string,paramName:string){    
    if(paramName == null || paramName == ""){
      for (let ops of this.listaParametros_opc){
        if (ops == name){
          return ""
        }
      }
      if (this.celdas_faltantes == '') {
        this.celdas_faltantes += name
      }else{
        this.celdas_faltantes += ','+name
      }
      this.isOkey = false
    }
    return paramName
  }

  clearData(){
    this.hayError = ''
    this.conexion.limpiarData()
    .subscribe(data => {
      let sucess = data.body.sucess
      if (sucess) {
        this.router.navigate(['/'])
      }
    },
    err => {
      console.log(err)
      this.hayError = `Error ${err.status} : ${err.message}`
    }
    )
  }

}
