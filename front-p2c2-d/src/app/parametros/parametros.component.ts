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

  constructor(private conexion:MainService,private router:Router) { }

  ngOnInit(): void {
    let data = this.conexion.obj_carga_to_parameter.data
    if (this.columnas.length == 0) { 
      console.log('no hay data')
      this.haydatos = false
    }else{
      this.columnas = data.columnas;
      this.haydatos = true
    }
    this.conexion.getDataParameter()
    .subscribe(data =>{
      console.log(data)
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
    },
    err => console.log(err)
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
    console.log('revisarArray',arr)
    if(arr == undefined){
      return []
    }else{
      return arr
    }
  }

}
