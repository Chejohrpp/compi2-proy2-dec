import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CargaFileComponent } from './carga-file.component';

describe('CargaFileComponent', () => {
  let component: CargaFileComponent;
  let fixture: ComponentFixture<CargaFileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CargaFileComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CargaFileComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
