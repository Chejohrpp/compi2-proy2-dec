import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MainService } from '../servicios/main.service';

@Component({
  selector: 'app-navegacion',
  templateUrl: './navegacion.component.html',
  styleUrls: ['./navegacion.component.css']
})
export class NavegacionComponent implements OnInit {
  private auxData:any
  constructor(private conexion:MainService,private router:Router) { }

  ngOnInit(): void {
    // console.log('obj navegacion',this.conexion.obj_carga_to_parameter)
    // this.conexion.carga_to_parameter.subscribe(data => {
    //   console.log('recibiendo data . . . . . ')
    //   console.log(data);
    // })
  }

}
