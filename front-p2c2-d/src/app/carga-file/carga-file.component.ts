import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MainService } from '../servicios/main.service';

@Component({
  selector: 'app-carga-file',
  templateUrl: './carga-file.component.html',
  styleUrls: ['./carga-file.component.css']
})
export class CargaFileComponent implements OnInit {

  file_entrada:any = null;
  nombre_archivo:string = 'Escoge un Archivo'
  isLoad:number = 0;
  hayError:string = '';

  constructor(private conexion:MainService,private router:Router) { } 

  ngOnInit(): void {
    // console.log('obj',this.conexion.obj_carga_to_parameter)
  }

  cambiarNombreInput(evento:any):void{
    this.nombre_archivo = evento.target.value.split( '\\' ).pop();
    this.file_entrada =  evento.target.files[0]
  }

  subirArchivo(form:any):void{
    if (this.file_entrada == null || this.file_entrada == '' || this.file_entrada == undefined) {
      this.isLoad = 2
    }else{
      this.isLoad = 1
      this.hayError = '';
      //hacer el proceso de enviar el dato
      this.conexion.sendFile(this.file_entrada)
      .subscribe(data =>{
        if (data.status == 200) {
          this.conexion.carga_to_parameter.emit({data:data.body});
          this.conexion.obj_carga_to_parameter = {data:data.body}
          //redirect
          this.router.navigate(['/parametrizacion'])
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

}
