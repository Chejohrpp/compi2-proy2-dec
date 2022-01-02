import { Component, OnInit,ViewChild,ElementRef } from '@angular/core';
import { MainService } from '../servicios/main.service';
import { jsPDF } from "jspdf";

@Component({
  selector: 'app-reporte',
  templateUrl: './reporte.component.html',
  styleUrls: ['./reporte.component.css']
})
export class ReporteComponent implements OnInit {

  @ViewChild('pdfd',{static: false}) el !: ElementRef;
  listaUrls:Array<string> = [];
  listaUrlsStatic:Array<string> = [];
  nombreReporte ='Reporte de Analisis';
  listaDataCalc = [];
  listaDataStatic = [];
  conclusion = '';
  baseUrl:string = ''
  hayData:boolean = false;

  constructor(private conexion:MainService) { }

  ngOnInit(): void {
    this.conexion.getDataReporte()
    .subscribe(data =>{
      // console.log(data);
      if (data.status === 200){
        // console.log('si hay data que se pueda mostar')
        this.hayData = true;
        this.listaUrls = data.body.lista_urls
        this.listaDataCalc = data.body.calculos_reporte
        this.listaUrlsStatic = data.body.lista_urls_static
        this.conclusion = data.body.conclusion
        this.listaDataStatic = data.body.datos_statics
        this.nombreReporte = data.body.name_case
        this.baseUrl = this.conexion.BASE_URL_PUBLIC
      }else{
        this.hayData = false;
      }
    },
    error => console.log(error)
    )
  }

  createPDF(){
    console.log('Create PDF')
    let pdf_d = new jsPDF('p','pt','a4');
    pdf_d.html(this.el.nativeElement,{
      'width': 180,
      //@ts-ignore
      pagesplit?: true,
      callback: (pdf_d) => {
        pdf_d.save("Reporte de analisis.pdf");
      }
    })
  }

}
