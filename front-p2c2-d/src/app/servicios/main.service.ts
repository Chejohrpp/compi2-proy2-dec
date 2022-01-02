import { EventEmitter, Injectable, Output } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MainService {
  @Output() carga_to_parameter: EventEmitter<any> = new EventEmitter();
  obj_carga_to_parameter:any = {};

  // private BASE_URL = 'http://localhost:4000';
  private BASE_URL = 'http://apic2p2dec-env.eba-gpkjep2y.us-east-2.elasticbeanstalk.com/api';
  BASE_URL_PUBLIC = 'http://apic2p2dec-env.eba-gpkjep2y.us-east-2.elasticbeanstalk.com'

  constructor(private http: HttpClient) { 
    
  }

  getHelloWorld(): Observable<any>{
    return this.http.get(`${this.BASE_URL}`)
  }

  sendFile(files:any): Observable<any>{
    const fd = new FormData();
    let file = <File> files
    fd.append('file_data',file,file.name);
    return this.http.post(`${this.BASE_URL}`,fd)
  }

  getDataParameter(): Observable<any>{
    return this.http.get(`${this.BASE_URL}/parametros`)
  }

  sendParameters(params:any): Observable<any>{
    return this.http.post(`${this.BASE_URL}/parametros`,params)
  }

  getDataReporte(): Observable<any>{
    return this.http.get(`${this.BASE_URL}/reporte`)
  }

  limpiarData(): Observable<any>{
    return this.http.get(`${this.BASE_URL}/clearall`)
  }
  
}
