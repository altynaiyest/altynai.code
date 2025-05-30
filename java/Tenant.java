package org.example.eeend.client;

import java.io.Serializable;

public class Tenant implements Serializable {
    private String fullName;
    private String passportNumber;
    private String phoneNumber;
    private String email;
    private double income;
    private boolean hasPets;
    private int rentalDurationMonths;

    public Tenant(String fullName, String passportNumber, String phoneNumber, String email,
                  double income, boolean hasPets, int rentalDurationMonths) {
        this.fullName = fullName;
        this.passportNumber = passportNumber;
        this.phoneNumber = phoneNumber;
        this.email = email;
        this.income = income;
        this.hasPets = hasPets;
        this.rentalDurationMonths = rentalDurationMonths;
    }

    public String getFullName() {
        return fullName;
    }

    public String getPassportNumber() {
        return passportNumber;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public String getEmail() {
        return email;
    }

    public double getIncome() {
        return income;
    }

    public boolean isHasPets() {
        return hasPets;
    }

    public int getRentalDurationMonths() {
        return rentalDurationMonths;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    public void setPassportNumber(String passportNumber) {
        this.passportNumber = passportNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public void setIncome(double income) {
        this.income = income;
    }

    public void setHasPets(boolean hasPets) {
        this.hasPets = hasPets;
    }

    public void setRentalDurationMonths(int rentalDurationMonths) {
        this.rentalDurationMonths = rentalDurationMonths;
    }

    // Полиморфизм — метод, который можно переопределять в наследниках
    public void displayInfo() {
        System.out.println(toString());
    }

    @Override
    public String toString() {
        return "Tenant {" +
                "Full Name='" + fullName + '\'' +
                ", Passport Number='" + passportNumber + '\'' +
                ", Phone Number='" + phoneNumber + '\'' +
                ", Email='" + email + '\'' +
                ", Income=" + income +
                ", Has Pets=" + hasPets +
                ", Rental Duration=" + rentalDurationMonths + " months" +
                '}';
    }
}
