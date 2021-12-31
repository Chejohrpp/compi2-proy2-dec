import { Component, OnInit } from '@angular/core';
import { MainService } from '../servicios/main.service';

@Component({
  selector: 'app-reporte',
  templateUrl: './reporte.component.html',
  styleUrls: ['./reporte.component.css']
})
export class ReporteComponent implements OnInit {

  constructor(private conexion:MainService) { }

  ngOnInit(): void {
    // this.conexion.getReporte()
    // .subscribe(data =>{
    //   console.log(data);
    // },
    // error => console.log(error)
    // )
  }

}
