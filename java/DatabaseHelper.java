package org.example.eeend.server;

import java.sql.*;
import org.example.eeend.client.Tenant;


public class DatabaseHelper {
    private static final String DB_URL = "jdbc:sqlite:database.db";

    static {
        try {
            Connection connection = DriverManager.getConnection(DB_URL);
            Statement stmt = connection.createStatement();
            String createTableSQL = "CREATE TABLE IF NOT EXISTS tenants (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "fullName TEXT, " +
                    "passportNumber TEXT, " +
                    "phoneNumber TEXT, " +
                    "email TEXT, " +
                    "income REAL, " +
                    "hasPets BOOLEAN, " +
                    "rentalDurationMonths INTEGER)";
            stmt.execute(createTableSQL);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static void saveTenantToDatabase(Tenant tenant) {
        String insertSQL = "INSERT INTO tenants (fullName, passportNumber, phoneNumber, email, income, hasPets, rentalDurationMonths) VALUES (?, ?, ?, ?, ?, ?, ?)";

        try (Connection connection = DriverManager.getConnection(DB_URL);
             PreparedStatement pstmt = connection.prepareStatement(insertSQL)) {
            pstmt.setString(1, tenant.getFullName());
            pstmt.setString(2, tenant.getPassportNumber());
            pstmt.setString(3, tenant.getPhoneNumber());
            pstmt.setString(4, tenant.getEmail());
            pstmt.setDouble(5, tenant.getIncome());
            pstmt.setBoolean(6, tenant.isHasPets());
            pstmt.setInt(7, tenant.getRentalDurationMonths());

            pstmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
